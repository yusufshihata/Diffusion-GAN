import torch

class DiagnosticDiffusionScheduler:
    def __init__(
        self,
        beta_0: float = 1e-4,
        beta_f: float = 2e-2,
        max_T: int = 1000,
        d_target: float = 0.6,
        smoothing: float = 0.999,
        C: float = 0.7
    ):
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.current_T = max_T
        self.max_T = max_T
        
        print(f"Initializing scheduler with max_T = {self.max_T}")
        
        # Generate initial schedules
        self.betas = self.get_betas_schedule(self.current_T)
        print(f"Created betas tensor with shape {self.betas.shape}")
        
        self.alphas = 1 - self.betas
        print(f"Created alphas tensor with shape {self.alphas.shape}")
        
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        print(f"Created alphas_cumprod tensor with shape {self.alphas_cumprod.shape}")
        
        self.d_target = d_target
        self.smoothing = smoothing
        self.r_d = d_target
        self.C = C
    
    def get_betas_schedule(self, length: int) -> torch.Tensor:
        print(f"Creating beta schedule with length {length}")
        return torch.linspace(self.beta_0, self.beta_f, length)
    
    def update_schedule(self, des_loss: torch.Tensor) -> None:
        print(f"Updating schedule with loss value: {des_loss}")
        
        if torch.is_tensor(des_loss):
            loss_value = des_loss.item() if des_loss.numel() == 1 else torch.mean(des_loss).item()
            sign_value = 1 if loss_value > 0 else -1
        else:
            loss_value = des_loss
            sign_value = 1 if loss_value > 0 else -1
            
        print(f"Loss value: {loss_value}, Sign value: {sign_value}")
        
        self.r_d = (1 - self.smoothing) * self.r_d + self.smoothing * sign_value
        print(f"Updated r_d to: {self.r_d}")
        
        # Calculate new T based on the ratio
        delta_T = int((1 if self.r_d > self.d_target else -1) * self.C)
        new_T = self.current_T + delta_T
        new_T = max(50, min(new_T, self.max_T))  # Keep T within valid range (minimum 50)
        
        print(f"Delta T: {delta_T}, New T: {new_T}, Current T: {self.current_T}")
        
        # Only recompute if T has changed
        if new_T != self.current_T:
            print(f"T changed from {self.current_T} to {new_T}. Recomputing schedules.")
            self.current_T = new_T
            self.betas = self.get_betas_schedule(self.current_T)
            self.alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            print(f"New alphas_cumprod shape: {self.alphas_cumprod.shape}")
        else:
            print("T remained unchanged. Keeping current schedules.")
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        print(f"Sampling {batch_size} timesteps from range [0, {self.current_T})")
        timesteps = torch.randint(0, self.current_T, (batch_size,), dtype=torch.long)
        print(f"Sampled timesteps shape: {timesteps.shape}, min: {timesteps.min()}, max: {timesteps.max()}")
        return timesteps
    
    def apply_diffusion(self, imgs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        print("\n--- APPLY DIFFUSION ---")
        print(f"imgs shape: {imgs.shape}")
        print(f"timesteps shape: {timesteps.shape}")
        print(f"timesteps min: {timesteps.min()}, max: {timesteps.max()}")
        print(f"alphas_cumprod shape: {self.alphas_cumprod.shape}")
        print(f"Current T value: {self.current_T}")
        
        # Check if any timesteps are out of bounds
        if torch.any(timesteps >= len(self.alphas_cumprod)):
            max_t = timesteps.max().item()
            print(f"WARNING: Invalid timestep! Max timestep: {max_t}, alphas_cumprod length: {len(self.alphas_cumprod)}")
            timesteps = torch.clamp(timesteps, 0, len(self.alphas_cumprod) - 1)
            print(f"Clamped timesteps min: {timesteps.min()}, max: {timesteps.max()}")
        
        # Get alpha values for each timestep
        try:
            print(f"Indexing alphas_cumprod with timesteps...")
            print(f"alphas_cumprod tensor device: {self.alphas_cumprod.device}")
            print(f"timesteps tensor device: {timesteps.device}")
            
            # Try moving tensors to same device if needed
            if self.alphas_cumprod.device != timesteps.device:
                print(f"Moving alphas_cumprod to {timesteps.device}")
                self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
            
            alphas_cumprod_t = self.alphas_cumprod[timesteps]
            print(f"alphas_cumprod_t shape after indexing: {alphas_cumprod_t.shape}")
            
            # Reshape for broadcasting
            alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1, 1)
            print(f"reshaped alphas_cumprod_t shape: {alphas_cumprod_t.shape}")
            
            # Generate noise
            noise = torch.randn_like(imgs)
            print(f"noise shape: {noise.shape}")
            
            # Apply noise based on the diffusion equation
            sqrt_alpha = torch.sqrt(alphas_cumprod_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod_t)
            
            print(f"sqrt_alpha shape: {sqrt_alpha.shape}")
            print(f"sqrt_one_minus_alpha shape: {sqrt_one_minus_alpha.shape}")
            print(f"imgs shape before multiplication: {imgs.shape}")
            
            term1 = sqrt_alpha * imgs
            print(f"term1 shape after sqrt_alpha * imgs: {term1.shape}")
            
            term2 = sqrt_one_minus_alpha * noise
            print(f"term2 shape after sqrt_one_minus_alpha * noise: {term2.shape}")
            
            noised_imgs = term1 + term2
            print(f"noised_imgs final shape: {noised_imgs.shape}")
            print("--- END APPLY DIFFUSION ---\n")
            
            return noised_imgs
            
        except Exception as e:
            print(f"Exception in apply_diffusion: {e}")
            # Print complete shapes of all tensors for debugging
            print(f"Complete imgs shape: {imgs.shape}")
            print(f"Complete timesteps shape and values: {timesteps.shape}, {timesteps}")
            print(f"Complete alphas_cumprod shape: {self.alphas_cumprod.shape}")
            raise