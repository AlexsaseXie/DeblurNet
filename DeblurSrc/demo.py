import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from ssim import SSIM
from PIL import Image
import util.util as util
import numpy as np
import torchvision.transforms as transforms


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def demo(image_path):

    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    img = np.array(img)
    result_img = np.zeros((img_height, img_width, 3), int)
    area_count = np.zeros((img_height, img_width), int)

    img_s = opt.fineSize1
    step = opt.fineSize1 // 2

    h_s = 0
    while (h_s < img_height):
        w_s = 0

        if (h_s + img_s > img_height):
            h_s = img_height - img_s

        while (w_s < img_width):
            if (w_s + img_s > img_width):
                w_s = img_width - img_s

            area_count[h_s:h_s + img_s, w_s:w_s + img_s] += 1

            img_patch1 = img[h_s:h_s + img_s, w_s:w_s + img_s, :]
            A_img1 = Image.fromarray(img_patch1)
            A_img2 = A_img1.resize((opt.fineSize2, opt.fineSize2), Image.ANTIALIAS)
            A_img3 = A_img2.resize((opt.fineSize3, opt.fineSize3), Image.ANTIALIAS)
            A_img1 = transformer(A_img1).unsqueeze(0)
            A_img2 = transformer(A_img2).unsqueeze(0)
            A_img3 = transformer(A_img3).unsqueeze(0)

            model.set_input({'A1': A_img1, 'A2': A_img2, 'A3': A_img3, 'A_paths': image_path,
                             'B1': A_img1, 'B2': A_img2, 'B3': A_img3})
            model.test()
            visuals = model.get_current_visuals()

            result_img[h_s:h_s + img_s, w_s:w_s + img_s, :] += visuals['Restored_Train'].astype(int)

            if (w_s + img_s >= img_width):
                break
            w_s += step

        if (h_s + img_s >= img_height):
            break
        h_s += step

    result_img[:, :, 0] = result_img[:, :, 0] / area_count
    result_img[:, :, 1] = result_img[:, :, 1] / area_count
    result_img[:, :, 2] = result_img[:, :, 2] / area_count

    result_img = Image.fromarray(result_img.astype(np.uint8))
    result_img.save('result.jpg')

    return 0, 0

demo(opt.image_path)
