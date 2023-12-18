from skimage import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import CONFIG


device = CONFIG.device


class SRData(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        img = io.imread(self.dataset_list[idx])
        img = self.transform(img)
        img = torch.div(img, 255)
        return img, self.getLabel(idx)

    def getLabel(self, idx):
        file = self.dataset_list[idx]
        filename = file.split('/')[-1]
        label = int(filename.split('_')[0])
        return label
