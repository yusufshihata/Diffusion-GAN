import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from debug.diffusion_scheduler_diagnosis import DiagnosticDiffusionScheduler

def diagnose_train(
        generator: nn.Module,
        discriminator: nn.Module,
        Goptimizer: optim.Optimizer,
        Doptimizer: optim.Optimizer,
        Gcriterion: nn.Module,
        Dcriterion: nn.Module,
        trainloader: DataLoader,
        epochs: int = 1,  # Run fewer epochs for debugging
        latent_dim: int = 784
) -> None:
    generator.train()
    discriminator.train()
    
    # Use diagnostic scheduler
    noise_scheduler = DiagnosticDiffusionScheduler(max_T=50)  # Lower max_T for debugging
    
    # Get sample batch to inspect shapes
    for real_img, _ in trainloader:
        print(f"Sample batch shape from dataloader: {real_img.shape}")
        break
    
    # Print model architecture for debugging
    print("\nGenerator architecture:")
    print(generator)
    print("\nDiscriminator architecture:")
    print(discriminator)
    
    print("\nStarting diagnostic training loop...")
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            for batch_idx, (real_img, _) in enumerate(trainloader):
                print(f"\nBatch {batch_idx+1}, Real image shape: {real_img.shape}")
                
                # Only process a few batches for debugging
                if batch_idx >= 3:
                    print("Processed 3 batches, stopping debug run")
                    return
                
                batch_size = real_img.shape[0]
                
                # Generate latent noise
                latent_noise = torch.randn((batch_size, 784), device=real_img.device)
                print(f"Latent noise shape: {latent_noise.shape}")
                
                # Generate fake images
                fake_img = generator(latent_noise)
                print(f"Generated fake image shape: {fake_img.shape}")
                
                # Sample timesteps for this batch
                timesteps = noise_scheduler.sample_timesteps(batch_size)
                
                # Apply diffusion to real and fake images
                try:
                    print("Applying diffusion to fake images...")
                    noised_fake_img = noise_scheduler.apply_diffusion(fake_img.detach(), timesteps)
                    
                    print("Applying diffusion to real images...")
                    noised_real_img = noise_scheduler.apply_diffusion(real_img, timesteps)
                    
                    print(f"Noised fake image shape: {noised_fake_img.shape}")
                    print(f"Noised real image shape: {noised_real_img.shape}")
                    
                    # Skip the rest of training for debugging purposes
                    print("Successfully applied diffusion to both real and fake images.")
                    break
                
                except Exception as e:
                    print(f"Exception in diffusion step: {e}")
                    # Try to diagnose the exact problem
                    if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                        # This is the same error we've been seeing
                        error_parts = str(e).split()
                        a_size = [int(s) for s in error_parts if s.isdigit()][0]
                        b_size = [int(s) for s in error_parts if s.isdigit()][1]
                        
                        print(f"\nDETAILED ERROR ANALYSIS:")
                        print(f"Tensor A size: {a_size}, Tensor B size: {b_size}")
                        print(f"Real image batch size: {real_img.shape[0]}")
                        print(f"Timesteps shape: {timesteps.shape}")
                        print(f"Unique timestep values: {torch.unique(timesteps)}")
                        print(f"alphas_cumprod length: {len(noise_scheduler.alphas_cumprod)}")
                        
                        # Try alternative approach - use same timestep for entire batch
                        print("\nTrying with constant timestep for whole batch...")
                        single_t = torch.tensor([10], device=real_img.device)
                        constant_t = single_t.repeat(batch_size)
                        print(f"Constant timesteps shape: {constant_t.shape}")
                        
                        try:
                            noised_fake_img = noise_scheduler.apply_diffusion(fake_img.detach(), constant_t)
                            print("Success with constant timestep approach!")
                        except Exception as e2:
                            print(f"Still failed with constant timestep: {e2}")
                    
                    return  # Stop after error for debugging
    
    except Exception as e:
        print(f"Exception in training loop: {e}")