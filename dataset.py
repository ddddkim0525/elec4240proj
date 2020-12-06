import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
class TextureDataset(Dataset):
    def __init__(self, diff_dir, disp_dir, nor_dir, rough_dir,transform=None):
        self.diff = diff_dir
        self.disp = disp_dir
        self.nor = nor_dir
        self.rough= rough_dir
        self.transform = transform
    def __len__(self):
        return len(os.listdir(self.diff))
    def __getitem__(self,idx):
        img_name = os.listdir(self.diff)[idx][:-11]
        diff_img = Image.open(self.diff+"/"+img_name+"diff_1k.jpg").convert('RGB')
        disp_img = Image.open(self.disp+"/"+img_name+"disp_1k.jpg").convert('L')
        nor_img = Image.open(self.nor+"/"+img_name+"nor_1k.jpg")
        rough_img = Image.open(self.rough+"/"+img_name+"rough_1k.jpg").convert('L')
        img_list = [diff_img, disp_img, nor_img, rough_img]
        seed = np.random.randint(2147483647)

        if self.transform is not None:
            for i, img in enumerate(img_list):
                random.seed(seed)
                torch.manual_seed(seed)
                img_list[i] = self.transform(img)

        return {"diff": img_list[0], "disp": img_list[1], "nor": img_list[2], "rough":img_list[3]}

data_transform = transforms.Compose([
    transforms.RandomCrop(900),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(64),
    transforms.ToTensor(),
])
diff_dir = "./Data/diff"
disp_dir = "./Data/disp"
nor_dir = "./Data/nor"
rough_dir = "./Data/rough"

dataset = TextureDataset(diff_dir, disp_dir, nor_dir, rough_dir, transform = data_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle =True)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
