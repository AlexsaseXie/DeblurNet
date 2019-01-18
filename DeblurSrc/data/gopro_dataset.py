import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np


class GoProDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = os.path.join(opt.dataroot)

        self.splits = os.listdir(self.root)

        self.A_paths = []
        self.B_paths = []

        for sp in self.splits:
            current_path = os.path.join(self.root, sp)
            
            current_A_root = os.path.join(current_path, 'blur')
            current_B_root = os.path.join(current_path, 'sharp')

            A_list = set()
            for A in os.listdir(current_A_root):
                A_list.add(A)

            B_list = set()
            for B in os.listdir(current_B_root):
                B_list.add(B)

            C_list = A_list & B_list

            for C in C_list: 
                self.A_paths.append(os.path.join(current_A_root, C))
                self.B_paths.append(os.path.join(current_B_root, C))

        self.size = len(self.A_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.size]
        B_path = self.B_paths[index % self.size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.size

    def name(self):
        return 'GoProDataset'


class GoProMultiScaleDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = os.path.join(opt.dataroot)

        self.splits = os.listdir(self.root)

        self.A_paths = []
        self.B_paths = []

        for sp in self.splits:
            current_path = os.path.join(self.root, sp)

            current_A_root = os.path.join(current_path, 'blur')
            current_B_root = os.path.join(current_path, 'sharp')

            A_list = set()
            for A in os.listdir(current_A_root):
                A_list.add(A)

            B_list = set()
            for B in os.listdir(current_B_root):
                B_list.add(B)

            C_list = A_list & B_list

            for C in C_list:
                self.A_paths.append(os.path.join(current_A_root, C))
                self.B_paths.append(os.path.join(current_B_root, C))

        self.size = len(self.A_paths)
        self.transformer = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.size]
        B_path = self.B_paths[index % self.size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.resize_or_crop == 'resize_and_crop':
            # resize
            A_img = np.array(A_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.ANTIALIAS))
            B_img = np.array(B_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.ANTIALIAS))

            # random crop
            beginX = np.random.randint(low=0, high=self.opt.loadSizeX - self.opt.fineSize1)
            beginY = np.random.randint(low=0, high=self.opt.loadSizeY - self.opt.fineSize1)
            A_img = A_img[beginY:beginY + self.opt.fineSize1, beginX:beginX + self.opt.fineSize1, :]
            B_img = B_img[beginY:beginY + self.opt.fineSize1, beginX:beginX + self.opt.fineSize1, :]

        else:
            A_img = np.array(A_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.ANTIALIAS))
            B_img = np.array(B_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.ANTIALIAS))
            A_img = A_img[self.opt.loadSizeY / 2 - self.opt.fineSize1 / 2: self.opt.loadSizeY / 2 + self.opt.fineSize1 / 2,
                    self.opt.loadSizeX / 2 - self.opt.fineSize1 / 2: self.opt.loadSizeX / 2 + self.opt.fineSize1 / 2, :]
            B_img = B_img[self.opt.loadSizeY / 2 - self.opt.fineSize1 / 2: self.opt.loadSizeY + self.opt.fineSize1 / 2,
                    self.opt.loadSizeX / 2 - self.opt.fineSize1 / 2: self.opt.loadSizeX / 2 + self.opt.fineSize1 / 2, :]

        # down scale
        A_img1 = Image.fromarray(A_img)
        A_img2 = A_img1.resize((self.opt.fineSize2, self.opt.fineSize2), Image.ANTIALIAS)
        A_img3 = A_img2.resize((self.opt.fineSize3, self.opt.fineSize3), Image.ANTIALIAS)

        B_img1 = Image.fromarray(B_img)
        B_img2 = B_img1.resize((self.opt.fineSize2, self.opt.fineSize2), Image.ANTIALIAS)
        B_img3 = B_img2.resize((self.opt.fineSize3, self.opt.fineSize3), Image.ANTIALIAS)

        # to tensor and normalize
        A_img1 = self.transformer(A_img1)
        A_img2 = self.transformer(A_img2)
        A_img3 = self.transformer(A_img3)

        B_img1 = self.transformer(B_img1)
        B_img2 = self.transformer(B_img2)
        B_img3 = self.transformer(B_img3)

        return {'A1': A_img1, 'A2': A_img2, 'A3': A_img3,
                'B1': B_img1, 'B2': B_img2, 'B3': B_img3,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.size

    def name(self):
        return 'GoProMultiScaleDataset'
