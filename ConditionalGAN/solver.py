import os
import torch
import torch.nn as nn
from torch import optim
from data_loader import get_loader
import torchvision.utils as vutils
from model import Generator, Discriminator


class Solver(object):
	def __init__(self, args):
		self.args = args

		# Get data loaders
		self.train_loader = get_loader(args)

		# Define Generator and Discriminator
		self.gen = Generator(z_dim=self.args.z_dim, n_classes=self.args.n_classes, label_embed_dim=self.args.label_embed_dim, n_channels=self.args.n_channels)
		self.dis = Discriminator(n_classes=self.args.n_classes, n_channels=self.args.n_channels)

		# Display Generator and Discriminators
		print('--------Generator--------')
		print(self.gen)
		print('--------Discriminator--------')
		print(self.dis)

		# Option to load pretrained model
		if self.args.load_model:
			print("Using pretrained model")
			self.gen.load_state_dict(torch.load(os.path.join(self.args.model_path, 'gen.pt')))
			self.dis.load_state_dict(torch.load(os.path.join(self.args.model_path, 'dis.pt')))

		# Training loss function
		if self.args.training_loss == 'bce':
			self.loss_fn = nn.BCEWithLogitsLoss()
		elif self.args.training_loss == 'mse':
			self.loss_fn = nn.MSELoss()

		# Fixed noise for tracking image generation across epochs
		self.fixed_z = torch.randn(self.args.n_images_to_display_per_class * self.args.n_classes, self.args.z_dim)
		self.fixed_y = torch.arange(0, self.args.n_classes)
		self.fixed_y = torch.repeat_interleave(self.fixed_y, self.args.n_images_to_display_per_class)

		# Push to GPU
		if self.args.is_cuda:
			self.gen     = self.gen.cuda()
			self.dis     = self.dis.cuda()
			self.fixed_z = self.fixed_z.cuda()
			self.fixed_y = self.fixed_y.cuda()

	def generate_images(self, name='final.png'):
		self.gen.eval()
		x_fake  = self.gen(self.fixed_z, self.fixed_y)
		x_fake  = (x_fake + 1) / 2
		x_fake_ = vutils.make_grid(x_fake, normalize=False, nrow=self.args.n_images_to_display_per_class)
		vutils.save_image(x_fake_, os.path.join(self.args.output_path, name))

	def train(self):
		iters_per_epoch = len(self.train_loader)

		# Define optimizer for training the model
		g_opt = optim.Adam(self.gen.parameters(), lr=self.args.lr, betas=(0.5, 0.999), weight_decay=2e-5)
		d_opt = optim.Adam(self.dis.parameters(), lr=self.args.lr, betas=(0.5, 0.999), weight_decay=2e-5)

		self.generate_images(name=f'sample_0.png')                       # Untrained model's generated image.

		# Training loop
		for epoch in range(self.args.epochs):
			# Set models to training mode
			self.gen.train()
			self.dis.train()

			# Loop on loader
			for i, (x, y) in enumerate(self.train_loader):

				z = torch.randn(self.args.batch_size, self.args.z_dim)

				# Push to GPU
				if self.args.is_cuda:
					x, y, z = x.cuda(), y.cuda(), z.cuda()

				# Generate fake data
				x_fake = self.gen(z, y)

				# Train Discriminator
				fake_out = self.dis(x_fake.detach(), y)
				real_out = self.dis(x, y)
				d_loss   = self.loss_fn(fake_out, torch.zeros_like(fake_out)) + self.loss_fn(real_out, torch.ones_like(fake_out))
				d_loss  /= 2

				d_opt.zero_grad()
				d_loss.backward()
				d_opt.step()

				# Train Generator
				fake_out = self.dis(x_fake, y)
				g_loss   = self.loss_fn(fake_out, torch.ones_like(fake_out))

				g_opt.zero_grad()
				g_loss.backward()
				g_opt.step()

				# Log training progress
				if i % 50 == 0 or i == (iters_per_epoch - 1):
					print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tdis_loss: {d_loss.item():.2f}\tgen_loss: {g_loss.item():.2f}')

			# Generate Image
			self.generate_images(name=f'sample_{epoch+1}.png')

			# Save model
			torch.save(self.gen.state_dict(), os.path.join(self.args.model_path, "gen.pt"))
			torch.save(self.dis.state_dict(), os.path.join(self.args.model_path, "dis.pt"))
