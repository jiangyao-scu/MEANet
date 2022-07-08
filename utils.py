import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import scipy.io as sio
import cv2
import torch


def crop(allfocus, fs, depth, gt, contour):
    index_x = 2 * random.randint(0, 9)
    index_y = 2 * random.randint(0, 9)
    new_allfocus = allfocus[index_x:index_x + 236, index_y:index_y + 236]
    new_depth = depth[index_x:index_x + 236, index_y:index_y + 236]
    new_gt = gt[index_x:index_x + 236, index_y:index_y + 236]
    new_contour = contour[index_x:index_x + 236, index_y:index_y + 236]
    new_fs = fs[index_x:index_x + 236, index_y:index_y + 236]

    new_allfocus = cv2.resize(new_allfocus, (256, 256))
    new_fs = cv2.resize(new_fs, (256, 256))
    new_depth = cv2.resize(new_depth, (256, 256))
    new_gt = cv2.resize(new_gt, (256, 256))
    new_contour = cv2.resize(new_contour, (256, 256))

    return new_allfocus, new_fs, new_depth, new_gt, new_contour


class ALLDataset(Dataset):
    def __init__(self, transform=True, location=None, train=True, crop=True):
        self.transform = transform
        self.mean_rgb = np.array([0.447, 0.407, 0.386])
        self.std_rgb = np.array([0.244, 0.250, 0.253])
        self.mean_focal = np.tile(self.mean_rgb, 12)
        self.std_focal = np.tile(self.std_rgb, 12)

        self.location = location
        self.num = len(os.listdir(self.location + 'depth/'))
        self.train = train
        self.crop = crop

    def mytransform(self, allfocus, fs, depth, gt, contour):
        allfocus = allfocus.astype(np.float32) / 255.0
        allfocus -= self.mean_rgb
        allfocus /= self.std_rgb

        fs = fs.astype(np.float32) / 255.0
        fs -= self.mean_focal
        fs /= self.std_focal

        depth = depth.astype(np.float32) / 255.0
        depth -= self.mean_rgb
        depth /= self.std_rgb

        gt = gt.astype(np.float32) / 255.0

        contour = contour.astype(np.float32) / 255.0

        return allfocus, fs, depth, gt, contour

    def mytransform_test(self, allfocus, fs, depth):
        allfocus = allfocus.astype(np.float32) / 255.0
        allfocus -= self.mean_rgb
        allfocus /= self.std_rgb

        fs = fs.astype(np.float32) / 255.0
        fs -= self.mean_focal
        fs /= self.std_focal

        depth = depth.astype(np.float32) / 255.0
        depth -= self.mean_rgb
        depth /= self.std_rgb

        return allfocus, fs, depth

    def __len__(self):
        return len(os.listdir(self.location + 'depth/'))

    def __getitem__(self, idx):
        img_name = os.listdir(self.location + 'depth/')[idx]
        allfocus = Image.open(self.location + 'allfocus/' + img_name.split('.')[0] + '.jpg')
        allfocus = allfocus.convert('RGB')
        allfocus = allfocus.resize((256, 256))
        allfocus = np.asarray(allfocus)

        depth = Image.open(self.location + 'depth/' + img_name.split('.')[0] + '.png')
        depth = depth.convert('RGB')
        depth = depth.resize((256, 256))
        depth = np.asarray(depth)

        focalstack = sio.loadmat(self.location + 'mat/' + img_name.split('.')[0] + '.mat')
        focal = focalstack['img']
        focal = np.asarray(focal, dtype=np.float32)

        if self.train:
            GT = Image.open(self.location + 'GT/' + img_name.split('.')[0] + '.png')
            GT = GT.convert('L')
            GT = GT.resize((256, 256))
            GT = np.asarray(GT)

            contour = Image.open(self.location + 'contour/' + img_name.split('.')[0] + '.png')
            contour = contour.convert('L')
            contour = contour.resize((256, 256))
            contour = np.asarray(contour)

            if self.crop:
                allfocus, focal, depth, GT, contour = crop(allfocus, focal, depth, GT, contour)
            if self.transform:
                allfocus, focal, depth, GT, contour = self.mytransform(allfocus, focal, depth, GT, contour)

            allfocus = transforms.ToTensor()(allfocus)
            depth = transforms.ToTensor()(depth)
            focal = transforms.ToTensor()(focal)
            GT = GT[..., np.newaxis]
            GT = transforms.ToTensor()(GT)
            contour = contour[..., np.newaxis]
            contour = transforms.ToTensor()(contour)
            return allfocus, depth, focal, GT, contour, img_name
        else:
            if self.transform:
                allfocus, focal, depth = self.mytransform_test(allfocus, focal, depth)
            allfocus = transforms.ToTensor()(allfocus)
            depth = transforms.ToTensor()(depth)
            focal = transforms.ToTensor()(focal)
            return allfocus, depth, focal, img_name


if __name__ == '__main__':
    train_dataloader = DataLoader(ALLDataset(location='dataset/LFSOD/Train/'),
                                  batch_size=1, shuffle=True, num_workers=1)
    for index, (allfocus, depth, fs, GT, contour, names) in enumerate(train_dataloader):
        # print(index,names[0],allfocus.size(),depth.size(),fs.size(),GT.size(),contour.size())
        print(index, names[0])

