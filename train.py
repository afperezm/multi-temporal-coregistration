import os
import torch
import torch.utils.data as data

from data import ImageFolder
from framework import MyFrame
from loss import DiceBCELoss
from networks.dinknet import DinkNet34
from time import time

SHAPE = (1024, 1024)
ROOT = 'dataset/train/'
image_list = list(filter(lambda x: x.find('sat') != -1, os.listdir(ROOT)))
train_list = list(map(lambda x: x[:-8], image_list))
NAME = 'log01_dink34'
BATCH_SIZE_PER_CARD = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

solver = MyFrame(DinkNet34, DiceBCELoss, device, 2e-4)
if torch.cuda.device_count() > 0:
    batch_size = torch.cuda.device_count() * BATCH_SIZE_PER_CARD
else:
    batch_size = BATCH_SIZE_PER_CARD

dataset = ImageFolder(train_list, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

my_log = open('logs/' + NAME + '.log', 'w')
tic = time()
no_optimization = 0
total_epoch = 300
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=my_log)
    print('epoch:', epoch, '    time:', int(time() - tic), file=my_log)
    print('train_loss:', train_epoch_loss, file=my_log)
    print('SHAPE:', SHAPE, file=my_log)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optimization += 1
    else:
        no_optimization = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optimization > 6:
        print('early stop at %d epoch' % epoch, file=my_log)
        print('early stop at %d epoch' % epoch)
        break
    if no_optimization > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, my_log=my_log)
    my_log.flush()

print('Finish!', file=my_log)
print('Finish!')
my_log.close()
