import argparse
import json
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from codebase.datasets.deepglobe import RoadsDataset
from codebase.models.dlinknet import DLinkNet34
from codebase.utils import transforms
from codebase.utils.losses import DiceBCELoss
from codebase.utils.metrics import BinaryAccuracy
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from time import strftime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


PARAMS = None


class DLinkNetModel(LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = DLinkNet34(backbone='imagenet')
        self.criterion = DiceBCELoss()
        self.metric = BinaryAccuracy()

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/iou", accuracy, on_step=False, on_epoch=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss, accuracy = self.shared_step(batch)
    #
    #     self.log("valid/loss", loss, on_step=False, on_epoch=True)
    #     self.log("valid/iou", accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch[0], batch[1]

        preds_list = []

        for img in images:
            img90 = torch.rot90(img, k=1, dims=[1, 2])
            img1 = torch.stack((img, img90))
            img2 = torch.flip(img1, dims=[2])  # Vertical flip
            img3 = torch.concatenate((img1, img2))
            img4 = torch.flip(img3, dims=[3])  # Horizontal flip
            # img5 = img3.transpose(0, 3, 1, 2)
            # img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
            # img5 = V(torch.Tensor(img5).to(self.device))
            img5 = img3
            # img6 = img4.transpose(0, 3, 1, 2)
            # img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
            # img6 = V(torch.Tensor(img6).to(self.device))
            img6 = img4

            pred_a = self.model(img5)
            pred_b = self.model(img6)

            pred1 = pred_a + torch.flip(pred_b, dims=[3])  # Revert horizontal flip
            pred2 = pred1[:2] + torch.flip(pred1[2:], dims=[2])  # Revert vertical flip
            pred3 = pred2[0] + torch.flip(torch.flip(torch.rot90(pred2[1], k=1, dims=[1, 2]), dims=[1]), dims=[2])

            pred3[pred3 > 4.0] = 255
            pred3[pred3 <= 4.0] = 0

            pred3 = pred3 / 255.0
            pred3[pred3 >= 0.5] = 1.0
            pred3[pred3 < 0.5] = 0.0

            preds_list.append(pred3)

        preds = torch.stack(preds_list)

        loss = self.criterion(preds, masks)
        accuracy = self.metric(preds, masks)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/iou", accuracy, on_step=False, on_epoch=True)

    def shared_step(self, batch):
        image, mask = batch[0], batch[1]

        pred = self.model(image)

        loss = self.criterion(pred, mask)

        accuracy = self.metric(pred, mask)

        return loss, accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
                                                               verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"}}


def main():
    data_dir = PARAMS.data_dir
    results_dir = PARAMS.results_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    test_batch_size = 1 if batch_size // 4 == 0 else batch_size // 4
    learning_rate = PARAMS.learning_rate
    name = PARAMS.name
    ckpt_path = PARAMS.ckpt_path

    results_dir_root = os.path.dirname(results_dir.rstrip('/'))
    results_dir_name = os.path.basename(results_dir.rstrip('/'))

    exp_name = f"{name}-{strftime('%y%m%d')}-{strftime('%H%M%S')}"

    if not os.path.exists(os.path.join("results", exp_name)):
        os.makedirs(os.path.join("results", exp_name))

    # Dump program arguments
    with open(os.path.join("results", exp_name, "params.json"), "w") as f:
        json.dump(vars(PARAMS), f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = RoadsDataset(data_dir=data_dir,
                                 is_train=True,
                                 transform=Compose([transforms.RandomHSV(hue_shift_limit=(-30, 30),
                                                                         sat_shift_limit=(-5, 5),
                                                                         val_shift_limit=(-15, 15)),
                                                    transforms.RandomShiftScale(shift_limit=(-0.1, 0.1),
                                                                                scale_limit=(-0.1, 0.1),
                                                                                aspect_limit=(-0.1, 0.1)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomRotation(),
                                                    transforms.Normalize(),
                                                    transforms.ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = RoadsDataset(data_dir=data_dir,
                                is_train=False,
                                transform=Compose([transforms.Normalize(),
                                                   transforms.ToTensor()]))
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

    # Initialize model
    roads_model = DLinkNetModel(lr=learning_rate)

    # Initialize logger
    logger = TensorBoardLogger(save_dir=results_dir_root, name=results_dir_name, version=exp_name, sub_dir="logs")

    # Initialize callbacks
    early_stopping = EarlyStopping(monitor="train/loss", min_delta=0.00, patience=6, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize trainer
    trainer = pl.Trainer(logger=logger, callbacks=[early_stopping, lr_monitor], enable_progress_bar=False, max_epochs=epochs,
                         accelerator=device)

    # Perform training
    if ckpt_path is None:
        trainer.fit(model=roads_model, train_dataloaders=train_dataloader)

    # Perform evaluation
    trainer.test(model=roads_model, dataloaders=test_dataloader, ckpt_path=ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Binary masks de-shifter.")
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--results_dir", help="Results directory", default="./results/")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=300)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=0.0002)
    parser.add_argument("--name", help="Model name", default="dlinknet34")
    parser.add_argument("--ckpt_path", help="Checkpoint path")
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
