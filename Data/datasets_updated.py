import os

import numpy as np
import argparse
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2 as cv
import nibabel as nib


class GetDataset(data.Dataset):
    def __init__(self, df, img_dir, transform, dataset, fold=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.data = dataset

        if self.data == 'CXR':
            self.df = self.df[self.df['fold'] == fold]
            self.df = self.df.set_index("Image Index")
            self.PRED_LABEL = [
                'Atelectasis',
                'Cardiomegaly',
                'Effusion',
                'Infiltration',
                'Mass',
                'Nodule',
                'Pneumonia',
                'Pneumothorax',
                'Consolidation',
                'Edema',
                'Emphysema',
                'Fibrosis',
                'Pleural_Thickening',
                'Hernia']

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if self.data == 'RCT':
            img_path = self.img_dir + self.df.iloc[idx, 0]
            act_idx = self.df.iloc[idx,7]
            im = Image.open(img_path)
            im2 = np.array(im, dtype=object)
            im2 = im2.astype('uint8')
            if len(im2.shape) == 3:
                im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
            im2 = im2[:, int(im2.shape[1] / 2):]
            im2 = cv.resize(im2, (128, 128), interpolation=cv.INTER_AREA)
            im = Image.fromarray(im2)
            image = self.transform(im)
            label = self.df.iloc[idx, 2]
        elif self.data == 'OASIS':
            act_idx = self.df['Ind'].iloc[idx]
            img_path = self.img_dir + self.df.iloc[idx, 0]
            im = Image.open(img_path)
            image = self.transform(im)
            label = self.df['Label'].iloc[idx]
        elif self.data == 'CXR':
            image = Image.open(
                os.path.join(
                    self.img_dir,
                    self.df.index[idx]))
            image = image.convert('RGB')

            label = np.zeros(len(self.PRED_LABEL), dtype=int)
            for i in range(0, len(self.PRED_LABEL)):
                # can leave zero if zero, else make one
                if (self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.PRED_LABEL[i].strip()
                    ].iloc[idx].astype('int')

            if self.transform:
                image = self.transform(image)
            act_idx = self.df.iloc[idx, -1]
        return image, label, idx, act_idx