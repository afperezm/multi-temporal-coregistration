import glob
import os
import pickle as pkl
import pandas as pd

scores_paths = sorted(glob.glob('./dlinknet34-imagenet-gillam-all-season-fold-[0-9]*-[0-9]*-[0-9]*/scores_coreg.pkl'))

precision = []
recall = []
iou = []
fold = []

for scores_path in scores_paths:
    with open(scores_path, 'rb') as file:
        scores = pkl.load(file)
    precision.append(scores['Precision'].mean())
    recall.append(scores['Recall'].mean())
    iou.append(scores['IoU'].mean())
    fold.append(int(os.path.basename(os.path.split(scores_path)[0]).split('-')[6]))

df = pd.DataFrame(list(zip(fold, precision, recall, iou)), columns=['Fold', 'Precision', 'Recall', 'IoU'])

list(zip(df.groupby('Fold')['IoU'].mean() * 100, df.groupby('Fold')['IoU'].std() * 100))

list(zip(df.groupby('Fold')['Precision'].mean() * 100, df.groupby('Fold')['Precision'].std() * 100))

list(zip(df.groupby('Fold')['Recall'].mean() * 100, df.groupby('Fold')['Recall'].std() * 100))
