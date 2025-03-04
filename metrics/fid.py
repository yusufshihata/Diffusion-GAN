import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# === FID Calculator Functions ===


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        self.feature_extractor = nn.Sequential(*list(inception.children())[:8])
        self.feature_extractor.eval()

    def forward(self, x):
        return self.feature_extractor(x).flatten(start_dim=1)


def calculate_statistics(dataloader, model, device):
    features = []
    for img in dataloader:
        img = img.to(device)
        with torch.no_grad():
            feat = model(img)
        features.append(feat)
    features = torch.cat(features, dim=0)
    mean = torch.mean(features, dim=0)
    diff = features - mean.unsqueeze(0)
    cov = (diff.T @ diff) / (features.shape[0] - 1)
    return mean, cov


def sqrtm_torch(matrix):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_matrix = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    return sqrt_matrix


def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = torch.sum((mu1 - mu2) ** 2)
    cov_sqrt = sqrtm_torch(sigma1 @ sigma2)
    fid = diff + torch.trace(sigma1 + sigma2 - 2 * cov_sqrt)
    return fid.item()


def compute_fid_metric(generator, dataloader, device, latent_dim=100, batch_size=32):
    generator.eval()
    inception = InceptionV3FeatureExtractor().to(device)
    inception.eval()

    mu_real, sigma_real = calculate_statistics(dataloader, inception, device)

    fake_images = []
    for real_imgs, _ in dataloader:
        current_bs = real_imgs.size(0)
        with torch.no_grad():
            noise = torch.randn(current_bs, latent_dim, device=device)
            fake = generator(noise)
        fake_images.append(fake)
    fake_images = torch.cat(fake_images, dim=0)

    fake_dataset = TensorDataset(fake_images)
    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: torch.stack([item[0] for item in batch], dim=0),
    )
    mu_fake, sigma_fake = calculate_statistics(fake_dataloader, inception, device)

    fid_value = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    generator.train()
    return fid_value
