import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, __scale_width
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import cv2

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

        self.resize = transforms.Compose([
            transforms.Resize([opt.loadSizeX, opt.loadSizeY],Image.BICUBIC)
        ])

        self.scale_width = transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX))

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.size]
        B_path = self.B_paths[index % self.size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if not 'crop' in self.opt.resize_or_crop:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        else :
            if self.opt.resize_or_crop == 'resize_and_crop':
                A_img = self.resize(A_img)
                B_img = self.resize(B_img)
            elif self.opt.resize_or_crop == 'scale_width_and_crop':
                A_img = self.scale_width(A_img)
                B_img = self.scale_width(B_img)

            A_img = self.totensor(A_img)
            B_img = self.totensor(B_img)

            w = A_img.size(2)
            h = A_img.size(1)

            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

            A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
                        w_offset:w_offset + self.opt.fineSize]
            B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
                        w_offset:w_offset + self.opt.fineSize]

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.size

    def name(self):
        return 'GoProDataset'
