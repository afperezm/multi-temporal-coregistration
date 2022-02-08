import torch
import torch.nn as nn


class BinaryAccuracy(nn.Module):

    @staticmethod
    def jaccard_score(predictions, labels, smooth=1e-7, threshold=0.5):

        # threshold predictions
        predictions = (predictions > threshold).float()

        # Flatten labels and predictions
        predictions = predictions.view(-1)
        labels = labels.view(-1)

        # Intersection is equivalent to true positive count
        intersection = torch.sum(torch.abs(predictions * labels))
        # Union is the mutually inclusive area of all labels and predictions
        union = torch.sum((torch.abs(predictions) + torch.abs(labels))) - intersection

        # Compute loss value
        iou = (intersection + smooth) / (union + smooth)

        return iou

    def __call__(self, y_true, y_pred):
        a = self.jaccard_score(y_pred, y_true)
        return a
