import random
import PIL.ImageOps
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Config():
    training_dir = "D:/yuxuan_/Siamese Network/Dataset/Resnet50/training/"
    testing_dir = "D:/yuxuan_/Siamese Network/Dataset/Resnet50/testing/"
    targetlist_dir = "C:/Users/yuxuan/1Project/SiameseNetwork_jupy/targetlist/targetlist"
    phish_dir = "C:/Users/yuxuan/1Project/SiameseNetwork_jupy/argetlist/phish"
    train_batch_size = 64
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1] and img0_tuple[0] != img1_tuple[0]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
        img0 = transform(img0)
        img1 = transform(img1)
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class TestYOLO_Dataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]
        img = Image.open(img_tuple[0])
        img_url = img_tuple[0]
        img_name = img_url[img_url.rfind("\\") + 1: img_url.rfind(".")]
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
