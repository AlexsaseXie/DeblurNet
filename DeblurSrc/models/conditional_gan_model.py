import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, networks_rnn
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain

		# define tensors
		self.input_A1 = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize1, opt.fineSize1)
		self.input_A2 = self.Tensor(opt.batchSize, opt.input_nc,
									opt.fineSize2, opt.fineSize2)
		self.input_A3 = self.Tensor(opt.batchSize, opt.input_nc,
									opt.fineSize3, opt.fineSize3)
		self.input_B1 = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize1, opt.fineSize1)
		self.input_B2 = self.Tensor(opt.batchSize, opt.output_nc,
									opt.fineSize2, opt.fineSize2)
		self.input_B3 = self.Tensor(opt.batchSize, opt.output_nc,
									opt.fineSize3, opt.fineSize3)

		# load/define networks
		#Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = not opt.gan_type == 'wgan-gp'
		# self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
		# 							  opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)
		self.netG = networks_rnn.DeblurNetGeneratorReuse((opt.fineSize1, opt.fineSize1), 3, opt.batchSize)
		self.netG = self.netG.cuda()
		self.netG.train()

		if self.isTrain:
			use_sigmoid = opt.gan_type == 'gan'
			self.netD = networks.define_D(opt.output_nc, opt.ndf,
										  opt.which_model_netD,
										  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
			self.netD = self.netD.cuda()
			self.netD.train()
		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
												
			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		# print('---------- Networks initialized -------------')
		# networks.print_network(self.netG)
		# if self.isTrain:
		# 	networks.print_network(self.netD)
		# print('-----------------------------------------------')

	def set_input(self, input):
		input_A1 = input['A1']
		input_A2 = input['A2']
		input_A3 = input['A3']
		input_B1 = input['B1']
		input_B2 = input['B2']
		input_B3 = input['B3']
		self.input_A1.resize_(input_A1.size()).copy_(input_A1)
		self.input_A2.resize_(input_A2.size()).copy_(input_A2)
		self.input_A3.resize_(input_A3.size()).copy_(input_A3)
		self.input_B1.resize_(input_B1.size()).copy_(input_B1)
		self.input_B2.resize_(input_B2.size()).copy_(input_B2)
		self.input_B3.resize_(input_B3.size()).copy_(input_B3)
		self.image_paths = input['A_paths']

	def forward(self):
		self.real_A1 = Variable(self.input_A1)
		self.real_A2 = Variable(self.input_A2)
		self.real_A3 = Variable(self.input_A3)
		self.fake_B3, self.fake_B2, self.fake_B1 = self.netG.forward(self.real_A3, self.real_A2, self.real_A1)
		self.real_B1 = Variable(self.input_B1)
		self.real_B2 = Variable(self.input_B2)
		self.real_B3 = Variable(self.input_B3)

	# no backprop gradients
	def test(self):
		with torch.no_grad():
			self.real_A1 = Variable(self.input_A1)
			self.real_A2 = Variable(self.input_A2)
			self.real_A3 = Variable(self.input_A3)
			self.fake_B3, self.fake_B2, self.fake_B1 = self.netG.forward(self.real_A3, self.real_A2, self.real_A1)
			self.real_B1 = Variable(self.input_B1)
			self.real_B2 = Variable(self.input_B2)
			self.real_B3 = Variable(self.input_B3)
		# self.real_A = Variable(self.input_A, volatile=True)
		# self.fake_B = self.netG.forward(self.real_A)
		# self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.real_A1, self.fake_B1, self.real_B1) + \
					  self.discLoss.get_loss(self.netD, self.real_A2, self.fake_B2, self.real_B2) + \
					  self.discLoss.get_loss(self.netD, self.real_A3, self.fake_B3, self.real_B3)

		self.loss_D.backward(retain_graph=True)

	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A1, self.fake_B1) + \
						  self.discLoss.get_g_loss(self.netD, self.real_A2, self.fake_B2) + \
						  self.discLoss.get_g_loss(self.netD, self.real_A3, self.fake_B3)
		self.loss_G_Content = (self.contentLoss.get_loss(self.fake_B1, self.real_B1) +
							   self.contentLoss.get_loss(self.fake_B2, self.real_B2) +
							   self.contentLoss.get_loss(self.fake_B3, self.real_B3)) * self.opt.lambda_A

		self.loss_G = self.loss_G_GAN + self.loss_G_Content

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
							('G_L1', self.loss_G_Content.item()),
							('D_real+fake', self.loss_D.data.item())
							])

	def get_current_visuals(self):
		real_A1 = util.tensor2im(self.real_A1.data)
		fake_B1 = util.tensor2im(self.fake_B1.data)
		real_B1 = util.tensor2im(self.real_B1.data)
		return OrderedDict([('Blurred_Train', real_A1), ('Restored_Train', fake_B1), ('Sharp_Train', real_B1)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
