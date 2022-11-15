import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter
import math
# dataset/model/loss function
from dataLoader import data_loader
from selfholo import selfholo
import perceptualloss as perceptualloss

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--run_id', type=str, default='', help='Experiment name', required=True)
p.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
p.add_argument('--size_of_miniBatches', type=int, default=1, help='Size of minibatch')
p.add_argument('--lr', type=float, default=4e-4, help='learning rate of Holonet weights')

# parse arguments
opt = p.parse_args()
run_id = opt.run_id

# tensorboard setup and file naming
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
writer = SummaryWriter(f'runs/{run_id}_{time_str}')

device = torch.device('cuda')

# Image data for training
train_loader = data_loader(opt)

# Load models #
self_holo = selfholo().to(device)
self_holo.train()  # generator to be trained

# Loss function
loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
loss = loss.to(device)
mseloss = nn.MSELoss()
mseloss = mseloss.to(device)

# create optimizer
optvars = self_holo.parameters()
optimizer = optim.Adam(optvars, lr=opt.lr)

# Training loop #
for i in range(opt.num_epochs):

    for k, target in enumerate(train_loader):
        # get target image
        amp, depth, mask, ikk = target
        amp, depth, mask = amp.to(device), depth.to(device), mask.to(device)
        source = torch.cat([amp, depth], dim=-3)

        optimizer.zero_grad()

        ik = k + i * len(train_loader)

        holo, slm_amp, recon_field = self_holo(source, ikk)

        output_amp = 0.95 * recon_field.abs()
        output_amp = output_amp * mask
        output_amp = output_amp.repeat(1, 3, 1, 1)
        amp_i = amp * mask
        amp_i = amp_i.repeat(1, 3, 1, 1)
        loss_pe = loss(output_amp, amp_i)
        mse = mseloss(slm_amp.mean(), slm_amp)
        loss_val = loss_pe + 0.1*mse

        loss_val.backward()
        optimizer.step()

        # print and output to tensorboard
        print(f'iteration {ik}: {loss_val.item()}')

        with torch.no_grad():
            writer.add_scalar('Loss', loss_val, ik)

            if ik % 50 == 0:
                writer.add_image('amp', (amp[0, ...]), ik)
                writer.add_image('depth', (depth[0, ...]), ik)
                writer.add_image('output_amp', (output_amp[0, ...]), ik)
                # normalize SLM phase
                writer.add_image('SLM Phase', (holo[0, ...] + math.pi) / (2 * math.pi), ik)

    # save trained model
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(self_holo.state_dict(), f'checkpoints/{run_id}_{time_str}_{i+1}.pth')

# python ./src/train.py  --run_id=selfholo
