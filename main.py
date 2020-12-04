import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch.optim as optim

from network import *
from dataset import *


device = torch.device("cuda")

dataset = TextureDataset(diff_dir, disp_dir, nor_dir, rough_dir, transform = data_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle =True)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

criterion = nn.BCELoss()

lr = 0.0002
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

beta1 = 0.5
# Setup Adam optimizers for both G and D
optimizerD_disp = optim.Adam(netD_disp.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD_nor = optim.Adam(netD_nor.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD_rough = optim.Adam(netD_rough.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizers = [optimizerD_disp, optimizerD_nor, optimizerD_rough]

