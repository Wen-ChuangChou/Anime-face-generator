from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# from diffusion import GaussianDiffusion
import os
from sys import platform
# import trainer
from pathlib import Path
# from unet import Unet

"""arguments"""
if platform == "linux":
    data_path = "./data/Animeface"
elif platform == "win32":
    data_dir = "Data/Animeface"
    current_dir = os.getcwd()
    current_path = Path(current_dir)
    data_path = current_path.parents[2] / Path(data_dir)

IMG_SIZE = 64
batch_size = 32
train_num_steps = 50000        # total training steps
lr = 1e-3
grad_steps = 1
ema_decay = 0.995           # exponential moving average decay

channels = 16             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4, 8)

timesteps = 1000            # Number of steps (adding noise)
beta_schedule = 'linear'

model = Unet(
    dim=channels,
    dim_mults=dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size=IMG_SIZE,
    timesteps=timesteps,
    beta_schedule=beta_schedule
)

trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size=batch_size,
    train_lr=lr,
    train_num_steps=train_num_steps,
    gradient_accumulate_every=grad_steps,
    ema_decay=ema_decay,
    save_and_sample_every=1000
)

if __name__ == '__main__':
    trainer.train()
