from torch import optim
import os
import torchvision.utils as vutils
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
import pdb

# Arguments
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 100
EPOCHS = 500
Z_DIM = 100
CHANNELS = 3
N_CRITIC = 5
GRADIENT_PENALTY = 10
LOAD_MODEL = False

DB = 'LSUN_Bedroom'  # CelebA | LSUN_Church | LSUN_Bedroom


# Directories for storing model and output samples
model_path = os.path.join('./model', DB)
if not os.path.exists(model_path):
	os.makedirs(model_path)
samples_path = os.path.join('./samples', DB)
if not os.path.exists(samples_path):
	os.makedirs(samples_path)
db_path = os.path.join('./data', DB)
if not os.path.exists(samples_path):
	os.makedirs(samples_path)


# Method for storing generated images
def generate_imgs(z, epoch=0):
	generator.eval()
	fake_imgs = generator(z)
	fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(z.shape[0] ** 0.5))
	vutils.save_image(fake_imgs_, os.path.join(samples_path, 'sample_' + str(epoch) + '.png'))


# Data loaders
transform = transforms.Compose([transforms.Resize([64, 64]),
								transforms.ToTensor(),
								transforms.Normalize([0.5], [0.5])])
if DB == 'LSUN_Church':
	dataset = datasets.LSUN(db_path, classes=['church_outdoor_train'], transform=transform)
elif DB == 'LSUN_Bedroom':
	dataset = datasets.LSUN(db_path, classes=['bedroom_train'], transform=transform)
	samples_to_use = list(range(200000))  # Use only 200k samples
	dataset = torch.utils.data.Subset(dataset, samples_to_use)
elif DB == 'CelebA':
	dataset = datasets.CelebA(db_path, split='train', download=True, transform=transform)
elif DB == 'MNIST':
	dataset = datasets.MNIST(db_path, train=True, download=True, transform=transform)
else:
	print("Incorrect dataset")
	exit(0)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
										  drop_last=True)

# Fix images for viz
fixed_z = torch.randn(IMGS_TO_DISPLAY, Z_DIM)

# Labels
real_label = torch.ones(BATCH_SIZE)
fake_label = torch.zeros(BATCH_SIZE)


# Networks
def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
	module = []
	if transpose:
		module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
	else:
		module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
	if use_bn:
		module.append(nn.BatchNorm2d(c_out))
	return nn.Sequential(*module)


class Generator(nn.Module):
	def __init__(self, z_dim=100, channels=3, conv_dim=64):
		super(Generator, self).__init__()
		self.tconv1 = conv_block(z_dim, conv_dim * 8, pad=0, transpose=True)
		self.tconv2 = conv_block(conv_dim * 8, conv_dim * 4, transpose=True)
		self.tconv3 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
		self.tconv4 = conv_block(conv_dim * 2, conv_dim, transpose=True)
		self.tconv5 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				nn.init.constant_(m.bias.data, 0)

	def forward(self, x):
		x = x.reshape([x.shape[0], -1, 1, 1])
		x = F.relu(self.tconv1(x))
		x = F.relu(self.tconv2(x))
		x = F.relu(self.tconv3(x))
		x = F.relu(self.tconv4(x))
		x = torch.tanh(self.tconv5(x))
		return x


class Critic(nn.Module):
	def __init__(self, channels=3, conv_dim=64):
		super(Critic, self).__init__()
		self.conv1 = conv_block(channels, conv_dim, use_bn=False)
		self.conv2 = conv_block(conv_dim, conv_dim * 2)
		self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
		self.conv4 = conv_block(conv_dim * 4, conv_dim * 8)
		self.conv5 = conv_block(conv_dim * 8, 1, k_size=4, stride=1, pad=0, use_bn=False)

		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				nn.init.constant_(m.bias.data, 0)

	def forward(self, x):
		alpha = 0.2
		x = F.leaky_relu(self.conv1(x), alpha)
		x = F.leaky_relu(self.conv2(x), alpha)
		x = F.leaky_relu(self.conv3(x), alpha)
		x = F.leaky_relu(self.conv4(x), alpha)
		x = self.conv5(x)
		return x.squeeze()

def gradient_penalty(real, fake):
	m = real.shape[0]
	epsilon = torch.rand(m, 1, 1, 1)
	if is_cuda:
		epsilon = epsilon.cuda()
	
	interpolated_img = epsilon * real + (1-epsilon) * fake
	interpolated_out = critic(interpolated_img)

	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
							   grad_outputs=torch.ones(interpolated_out.shape).cuda() if is_cuda else torch.ones(interpolated_out.shape),
							   create_graph=True, retain_graph=True)[0]
	grads = grads.reshape([m, -1])
	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 
	return grad_penalty


generator = Generator(z_dim=Z_DIM, channels=CHANNELS)
critic = Critic(channels=CHANNELS)

# Load previous model   
if LOAD_MODEL:
	generator.load_state_dict(torch.load(os.path.join(model_path, 'generator.pkl')))
	critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pkl')))

# Model Summary
print("------------------Generator------------------")
print(generator)
print("------------------Critic------------------")
print(critic)

# Define Optimizers
g_opt = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9), weight_decay=2e-5)
c_opt = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9), weight_decay=2e-5)

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
	generator, critic = generator.cuda(), critic.cuda()
	real_label, fake_label = real_label.cuda(), fake_label.cuda()
	fixed_z = fixed_z.cuda()

total_iters = 0
g_loss = d_loss = torch.Tensor([0])
max_iter = len(data_loader)

# Training
for epoch in range(EPOCHS):
	generator.train()
	critic.train()

	for i, data in enumerate(data_loader):

		total_iters += 1

		# Loading data
		x_real, _ = data
		z_fake = torch.randn(BATCH_SIZE, Z_DIM)

		if is_cuda:
			x_real = x_real.cuda()
			z_fake = z_fake.cuda()

		# Generate fake data
		x_fake = generator(z_fake)

		# Train Critic
		fake_out = critic(x_fake.detach())
		real_out = critic(x_real.detach())
		x_out = torch.cat((real_out, fake_out))
		d_loss = -(real_out.mean() - fake_out.mean()) + gradient_penalty(x_real, x_fake) * GRADIENT_PENALTY + (x_out ** 2).mean() * 0.001

		c_opt.zero_grad()
		d_loss.backward()
		c_opt.step()

		# Train Generator
		if total_iters % N_CRITIC == 0:
			z_fake = torch.randn(BATCH_SIZE, Z_DIM)
			if is_cuda:
				z_fake = z_fake.cuda()
			x_fake = generator(z_fake)

			fake_out = critic(x_fake)
			g_loss = - fake_out.mean()

			g_opt.zero_grad()
			g_loss.backward()
			g_opt.step()

		if i % 50 == 0:
			print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
				  + "\titer: " + str(i) + "/" + str(max_iter)
				  + "\ttotal_iters: " + str(total_iters)
				  + "\td_loss:" + str(round(d_loss.item(), 4))
				  + "\tg_loss:" + str(round(g_loss.item(), 4))
				  )

	torch.save(generator.state_dict(), os.path.join(model_path, 'generator.pkl'))
	torch.save(critic.state_dict(), os.path.join(model_path, 'critic.pkl'))

	generate_imgs(fixed_z, epoch=epoch + 1)

generate_imgs(fixed_z)
