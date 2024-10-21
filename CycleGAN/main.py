from model import Discriminator, Generator
from data_loader import get_loaders
from utils import generate_imgs
from torch import optim
import torch
import os

EPOCHS = 300  # 50-300
BATCH_SIZE = 128
LOAD_MODEL = False

IMAGE_SIZE = 32
A_DS = 'svhn'
A_Channels = 3
B_DS = 'mnist'
B_Channels = 1

# Directories for storing model and output samples
model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
samples_path = './samples'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)
db_path = './data'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

# Networks
ab_gen = Generator(in_channels=A_Channels, out_channels=B_Channels)
ba_gen = Generator(in_channels=B_Channels, out_channels=A_Channels)
a_disc = Discriminator(channels=A_Channels)
b_disc = Discriminator(channels=B_Channels)

# Load previous model   
if LOAD_MODEL:
    ab_gen.load_state_dict(torch.load(os.path.join(model_path, 'ab_gen.pkl')))
    ba_gen.load_state_dict(torch.load(os.path.join(model_path, 'ba_gen.pkl')))
    a_disc.load_state_dict(torch.load(os.path.join(model_path, 'a_disc.pkl')))
    b_disc.load_state_dict(torch.load(os.path.join(model_path, 'b_disc.pkl')))

# Define Optimizers
g_opt = optim.Adam(list(ab_gen.parameters()) + list(ba_gen.parameters()), lr=0.0002, betas=(0.5, 0.999),
                   weight_decay=2e-5)
d_opt = optim.Adam(list(a_disc.parameters()) + list(b_disc.parameters()), lr=0.0002, betas=(0.5, 0.999),
                   weight_decay=2e-5)

# Data loaders
a_loader, b_loader = get_loaders(db_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, a_ds=A_DS, b_ds=B_DS)
iters_per_epoch = min(len(a_loader), len(b_loader))

# Fix images for viz
a_fixed = iter(a_loader).next()[0]
b_fixed = iter(b_loader).next()[0]

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    ab_gen, ba_gen = ab_gen.cuda(), ba_gen.cuda()
    a_disc, b_disc = a_disc.cuda(), b_disc.cuda()

    a_fixed = a_fixed.cuda()
    b_fixed = b_fixed.cuda()

# Cycle-GAN Training
for epoch in range(EPOCHS):
    ab_gen.train()
    ba_gen.train()
    a_disc.train()
    b_disc.train()

    for i, (a_data, b_data) in enumerate(zip(a_loader, b_loader)):

        # Loading data
        a_real, _ = a_data
        b_real, _ = b_data

        if is_cuda:
            a_real, b_real = a_real.cuda(), b_real.cuda()

        # Fake Images
        b_fake = ab_gen(a_real)
        a_fake = ba_gen(b_real)

        # Training discriminator
        a_real_out = a_disc(a_real)
        a_fake_out = a_disc(a_fake.detach())
        a_d_loss = (torch.mean((a_real_out - 1) ** 2) + torch.mean(a_fake_out ** 2)) / 2

        b_real_out = b_disc(b_real)
        b_fake_out = b_disc(b_fake.detach())
        b_d_loss = (torch.mean((b_real_out - 1) ** 2) + torch.mean(b_fake_out ** 2)) / 2

        d_opt.zero_grad()
        d_loss = a_d_loss + b_d_loss
        d_loss.backward()
        d_opt.step()

        # Training Generator
        a_fake_out = a_disc(a_fake)
        b_fake_out = b_disc(b_fake)

        a_g_loss = torch.mean((a_fake_out - 1) ** 2)
        b_g_loss = torch.mean((b_fake_out - 1) ** 2)
        g_gan_loss = a_g_loss + b_g_loss

        a_g_ctnt_loss = (a_real - ba_gen(b_fake)).abs().mean()
        b_g_ctnt_loss = (b_real - ab_gen(a_fake)).abs().mean()
        g_ctnt_loss = a_g_ctnt_loss + b_g_ctnt_loss

        g_opt.zero_grad()
        g_loss = g_gan_loss + g_ctnt_loss
        g_loss.backward()
        g_opt.step()

        if i % 50 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + " it: " + str(i) + "/" + str(iters_per_epoch)
                  + "\ta_d_loss:" + str(round(a_d_loss.item(), 4))
                  + "\ta_g_loss:" + str(round(a_g_loss.item(), 4))
                  + "\ta_g_ctnt_loss:" + str(round(a_g_ctnt_loss.item(), 4))
                  + "\tb_d_loss:" + str(round(b_d_loss.item(), 4))
                  + "\tb_g_loss:" + str(round(b_g_loss.item(), 4))
                  + "\tb_g_ctnt_loss:" + str(round(b_g_ctnt_loss.item(), 4)))

    torch.save(ab_gen.state_dict(), os.path.join(model_path, 'ab_gen.pkl'))
    torch.save(ba_gen.state_dict(), os.path.join(model_path, 'ba_gen.pkl'))
    torch.save(a_disc.state_dict(), os.path.join(model_path, 'a_disc.pkl'))
    torch.save(b_disc.state_dict(), os.path.join(model_path, 'b_disc.pkl'))

    generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path, epoch=epoch + 1)

generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen, samples_path)
