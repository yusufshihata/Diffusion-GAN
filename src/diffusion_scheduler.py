import torch


class AdaptiveDiffusionScheduler:
    def __init__(
        self,
        beta_0=1e-4,
        beta_f=2e-2,
        max_T=1000,
        d_target=0.6,
        smoothing=0.999,
        C=0.7,
        device="cuda",
    ):
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.T = max_T
        self.max_T = max_T
        self.device = device
        self.betas = torch.linspace(beta_0, beta_f, max_T).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.d_target = d_target
        self.smoothing = smoothing
        self.r_d = d_target
        self.C = C

    def update_schedule(self, des_loss: float) -> None:
        self.r_d = (1 - self.smoothing) * self.r_d + self.smoothing * des_loss
        self.T = int(self.T + self.C * (self.r_d - self.d_target))
        self.T = max(1, min(self.T, self.max_T))
        self.betas = torch.linspace(self.beta_0, self.beta_f, self.T).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        current_T = min(self.T, len(self.alphas_cumprod) - 1)
        return torch.randint(
            0, current_T, (batch_size,), dtype=torch.long, device=self.device
        )

    def apply_diffusion(
        self, imgs: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        timesteps = torch.clamp(timesteps, 0, len(self.alphas_cumprod) - 1)
        alphas_cumprod_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        noise = torch.randn_like(imgs)
        return (
            torch.sqrt(alphas_cumprod_t) * imgs
            + torch.sqrt(1 - alphas_cumprod_t) * noise
        )
