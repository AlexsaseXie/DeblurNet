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
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()
	if opt.dataset_mode != 'single':
		real_B = util.tensor2im(data['B'])

		avgPSNR += PSNR(visuals['fake_B'],real_B)
		pilFake = Image.fromarray(visuals['fake_B'])
		pilReal = Image.fromarray(real_B)
		avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	img_path = model.get_image_paths()
	print('process image... %s' % img_path)
	visualizer.save_images(webpage, visuals, img_path)

if opt.dataset_mode != 'single':	
	avgPSNR /= counter
	avgSSIM /= counter
	with open(os.path.join(opt.results_dir, opt.name, 'test_latest', 'result.txt'),'w') as f:
		f.write('PSNR = %f\n' % avgPSNR)
		f.write('SSIM = %f\n' % avgSSIM)
	print('PSNR = %f, SSIM = %f' %
				  (avgPSNR, avgSSIM))

webpage.save()
