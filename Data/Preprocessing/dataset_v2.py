import numpy as np
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2 as cv

# Good for quickly iterating through the dataset (like getting mean,std,etc.)
class GetDataset(data.Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        #self.transform = transform

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.img_dir + self.df.iloc[idx, 0]
        im = Image.open(img_path)
        im2 = np.array(im, dtype=object)
        im2 = im2.astype('uint8')
        if len(im2.shape) == 3:
            im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        #im2 = im2[:, int(im2.shape[1] / 2):] # uncomment this for RCT; not for OASIS
        #im2 = cv.resize(im2, (224, 224), interpolation=cv.INTER_AREA) # change to 128 x 128 for RCT
        # if idx == 0:
        #     plt.imsave('/home/zoe/HEYYYslice.png', im2, cmap='gray')
        image = im2 * 1.0
        return image