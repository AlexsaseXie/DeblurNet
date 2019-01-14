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
import numpy as np
from collections import OrderedDict
import util.util as util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    img_width = data['A'].shape[2]
    img_height = data['A'].shape[3]

    img = np.zeros((img_width, img_height, 3), int)
    area_count = np.zeros((img_width, img_height), int)

    w_s = 0
    h_s = 0
    
    img_s = opt.fineSize
    step = opt.fineSize // 2

    while(h_s < img_height) :
        w_s = 0

        if (h_s + img_s > img_height) :
            h_s = img_height - img_s

        while (w_s < img_width):
            if (w_s + img_s > img_width):
                w_s = img_width - img_s
            
            area_count[w_s:w_s+img_s, h_s:h_s+img_s] += 1

            model.set_input({'A': data['A'][:,:,w_s:w_s+img_s, h_s:h_s+img_s], 'A_paths': data['A_paths']})
            model.test()
            visuals = model.get_current_visuals()

            img[w_s:w_s+img_s, h_s:h_s+img_s, :] += visuals['fake_B'].astype(int)

            if (w_s + img_s >= img_width):
                break
            w_s += step

        if (h_s + img_s >= img_height):
            break
        h_s += step
        
    img[:,:,0] = img[:,:,0] / area_count
    img[:,:,1] = img[:,:,1] / area_count
    img[:,:,2] = img[:,:,2] / area_count

    fake_B = img.astype(np.uint8)
    if opt.dataset_mode != 'single':
        real_B = util.tensor2im(data['B'])

        avgPSNR += PSNR(fake_B,real_B)
        pilFake = Image.fromarray(fake_B)
        pilReal = Image.fromarray(real_B)
        avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)

    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, OrderedDict([('real_A', util.tensor2im(data['A'])),('fake_B', fake_B)]), img_path)
	
if opt.dataset_mode != 'single':	
    avgPSNR /= counter
    avgSSIM /= counter
    with open(os.path.join(opt.results_dir, opt.name, 'test_latest', 'result.txt'),'w') as f:
        f.write('PSNR = %f\n' % avgPSNR)
        f.write('SSIM = %f\n' % avgSSIM)
    print('PSNR = %f, SSIM = %f' %
    			  (avgPSNR, avgSSIM))


webpage.save()
