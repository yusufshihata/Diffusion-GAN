import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils.diffusion_scheduler import AdaptiveDiffusionScheduler
from utils.logger import Logger
from scripts.inference import inference
from scripts.utils import save_checkpoint
from metrics.fid import compute_fid_metric


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    Goptimizer: optim.Optimizer,
    Doptimizer: optim.Optimizer,
    Gcriterion: nn.Module,
    Dcriterion: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    latent_dim: int = 100,
    save_dirs: str = "models",
    checkpoint_interval: int = 1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device).train()
    discriminator.to(device).train()
    noise_scheduler = AdaptiveDiffusionScheduler(device=device)
    os.makedirs(save_dirs, exist_ok=True)

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for real_img, _ in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            real_img = real_img.to(device)
            batch_size = real_img.size(0)
            num_batches += 1

            timesteps = noise_scheduler.sample_timesteps(batch_size)
            latent_noise = torch.randn(batch_size, latent_dim, device=device)
            fake_img = generator(latent_noise)

            # --- Train Discriminator ---
            Doptimizer.zero_grad()
            noised_real_img = noise_scheduler.apply_diffusion(real_img, timesteps)
            noised_fake_img = noise_scheduler.apply_diffusion(
                fake_img.detach(), timesteps
            )
            real_outputs = discriminator(noised_real_img)
            fake_outputs = discriminator(noised_fake_img)
            d_loss = Dcriterion(
                discriminator,
                noised_real_img,
                noised_fake_img,
                real_outputs,
                fake_outputs,
            )
            d_loss.backward()
            Doptimizer.step()

            # --- Train Generator ---
            Goptimizer.zero_grad()
            noised_fake_img = noise_scheduler.apply_diffusion(fake_img, timesteps)
            fake_outputs = discriminator(noised_fake_img)
            g_loss = Gcriterion(fake_outputs)
            g_loss.backward()
            Goptimizer.step()

            noise_scheduler.update_schedule(d_loss.item())
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        inference(fake_img, 32, save_path=f"./output/batch_{epoch}.png")

        fid_score = compute_fid_metric(generator, trainloader, device, latent_dim)
        print(f"Epoch {epoch + 1}: FID Score: {fid_score:.4f}")

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                generator,
                discriminator,
                Goptimizer,
                Doptimizer,
                epoch=epoch + 1,
                save_dir=save_dirs,
            )

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        Logger.log(epoch, avg_g_loss, avg_d_loss)
        print(f"Epoch {epoch + 1}: G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}")
