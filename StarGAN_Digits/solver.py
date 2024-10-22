import os
import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torchvision.utils as vutils
from data_loader import get_loader
from model import Generator, Critic
from utils import gradient_penalty

class Solver(object):
	def __init__(self, args):
		self.args = args

		# Get data loaders
		self.train_loader = get_loader(self.args.datasets, self.args.data_path, self.args.batch_size, self.args.n_workers)

		# Define Generator and Critic
		self.gen = Generator(n_domains=self.args.n_domains)
		self.ctc = Critic(n_domains=self.args.n_domains)

		# Display Generator and Critics
		print('--------Generator--------')
		print(self.gen)
		print('--------Critic--------')
		print(self.ctc)

		# Option to load pretrained model
		if self.args.load_model:
			print("Using pretrained model")
			self.gen.load_state_dict(torch.load(os.path.join(self.args.model_path, 'gen.pt')))
			self.ctc.load_state_dict(torch.load(os.path.join(self.args.model_path, 'ctc.pt')))

		# Define domain classification Loss
		self.ce = nn.CrossEntropyLoss()
		self.l1 = nn.L1Loss()

		# Fixed images for tracking image generation across epochs
		self.fixed_image = next(iter(self.train_loader))[0][:self.args.n_images_to_display]

		# Push to GPU
		if self.args.is_cuda:
			self.gen         = self.gen.cuda()
			self.ctc         = self.ctc.cuda()
			self.fixed_image = self.fixed_image.cuda()

	def generate_images(self, name='final.png'):
		self.gen.eval()
		m = self.fixed_image.shape[0]

		y = torch.arange(start=-1, end=self.args.n_domains)
		y = y.expand(m, self.args.n_domains+1).reshape([-1])
		
		if self.args.is_cuda:
			y = y.cuda()
		x = torch.repeat_interleave(self.fixed_image, self.args.n_domains+1, dim=0)

		real_y = torch.arange(start=0, end=m*(self.args.n_domains+1), step=self.args.n_domains+1)   
		y[real_y] = 0

		display_imgs = self.gen(x, y)
		display_imgs[real_y] = self.fixed_image

		display_imgs = (display_imgs + 1)/2
		display_imgs.clamp_(0, 1)
		
		display_imgs = vutils.make_grid(display_imgs, normalize=False, nrow=self.args.n_domains+1, padding=2, pad_value=1)
		vutils.save_image(display_imgs, os.path.join(self.args.output_path, name))

	def generate_sample_images(self):
		x = next(iter(self.train_loader))[0]
		x = (x + 1) / 2
		x = vutils.make_grid(x, normalize=False, nrow=int(x.shape[0]**0.5))
		vutils.save_image(x, os.path.join(self.args.output_path, 'x_original.png'))

	def gradient_penalty(self, real, fake):
	    m = real.shape[0]
	    epsilon = torch.rand(m, 1, 1, 1)
	    if self.args.is_cuda:
	        epsilon = epsilon.cuda()
	    
	    interpolated_img = epsilon * real + (1-epsilon) * fake
	    interpolated_out = self.ctc(interpolated_img)[0]

	    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                              grad_outputs=torch.ones(interpolated_out.shape).cuda() if self.args.is_cuda else torch.ones(interpolated_out.shape),
                              create_graph=True, retain_graph=True)[0]
	    grads = grads.reshape([m, -1])
	    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 
	    return grad_penalty

	def train(self):
		iters_per_epoch = min(1000, len(self.train_loader))				# Set max 1000 iterations in one epoch

		# Define optimizer for training the model
		g_opt = optim.Adam(self.gen.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
		c_opt = optim.Adam(self.ctc.parameters(), lr=self.args.lr, betas=(0.5, 0.999))		

		self.generate_images(name=f'sample_0.png')                       # Untrained model's generated image.

		total_iter = 0
		# Training loop
		for epoch in range(self.args.epochs):
			# Set models to training mode
			self.gen.train()
			self.ctc.train()

			# Loop on loader
			for i, data in enumerate(self.train_loader):

				x, y = data

				# Push to GPU
				if self.args.is_cuda:
					x, y = x.cuda(), y.cuda()

				target_y = y[torch.randperm(y.shape[0])]

				# Generate fake data
				x_fake = self.gen(x, target_y)

				# Train Critic
				real_gan_out, real_cls_out = self.ctc(x)
				fake_gan_out, fake_cls_out = self.ctc(x_fake.detach())

				# d_gan_loss = -(real_gan_out.mean() - fake_gan_out.mean()) + self.gradient_penalty(x, x_fake, self.ctc, self.args.is_cuda) * self.args.gradient_penalty
				d_gan_loss = -(real_gan_out.mean() - fake_gan_out.mean()) + self.gradient_penalty(x, x_fake) * self.args.gradient_penalty
				d_clf_loss = self.ce(real_cls_out, y)

				x_out  = torch.cat((real_gan_out, fake_gan_out))
				out_loss = (x_out ** 2).mean() * 0.0001   					# To keep outputs close to 0

				c_opt.zero_grad()
				d_loss = d_gan_loss + d_clf_loss + out_loss
				d_loss.backward()
				c_opt.step()

				# Training Generator
				if total_iter % self.args.n_dis_per_gen_update == 0:
					x_fake = self.gen(x, target_y)
					fake_gan_out, fake_cls_out = self.ctc(x_fake)

					g_gan_loss = - fake_gan_out.mean()
					g_clf_loss = self.ce(fake_cls_out, target_y)
					g_rec_loss = self.l1(x, self.gen(x_fake, y))
					g_idt_loss = self.l1(x, self.gen(x, y))

					g_opt.zero_grad()
					g_loss = g_gan_loss + g_clf_loss + g_rec_loss + g_idt_loss
					g_loss.backward()
					g_opt.step()

				# Log training progress
				if i % 50 == 0 or i == (iters_per_epoch - 1):
					print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tdis_loss: {d_loss.item():.2f}\tgen_loss: {g_loss.item():.2f}')

				total_iter += 1

				if i == 999:
					break

			# Generate Image
			self.generate_images(name=f'sample_{epoch+1}.png')

			# Save model
			torch.save(self.gen.state_dict(), os.path.join(self.args.model_path, "gen.pt"))
			torch.save(self.ctc.state_dict(), os.path.join(self.args.model_path, "ctc.pt"))
