
import os, sys
import numpy as np

from torch.utils.data import Dataset
from PIL import Image  

class my_dataset(Dataset):

    def __init__(self, files_list, labels_list, transform, data_folder):
        self.files_list = [file.replace('/', os.sep).replace('\\', os.sep) for file in files_list]
        self.files_list = [os.path.join(data_folder, file) for file in self.files_list]
        assert all([os.path.exists(file) for file in self.files_list])

        self.labels_list = labels_list
        assert len(self.files_list) == len(self.labels_list)
        
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.files_list[index]).convert("RGB")
        img = self.transform(img)
        return img, self.labels_list[index]

    def __len__(self):
        return len(self.labels_list)