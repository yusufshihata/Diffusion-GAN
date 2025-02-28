import torch
import torch.nn as nn

class DiscriminatorLoss(nn.Module):
    def __init__(self, lambda_gp: float = 10.0):
        super(DiscriminatorLoss, self).__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.lambda_gp = lambda_gp
    
    def forward(self, 
                discriminator: nn.Module, 
                real_images: torch.Tensor, 
                fake_images: torch.Tensor, 
                real_outputs: torch.Tensor, 
                fake_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discriminator: The discriminator model (for gradient penalty computation)
            real_images: Batch of real images [batch_size, 1, 28, 28]
            fake_images: Batch of fake images [batch_size, 1, 28, 28]
            real_outputs: Discriminator logits for real images [batch_size, 1]
            fake_outputs: Discriminator logits for fake images [batch_size, 1]
        Returns:
            Total loss (BCE + gradient penalty)
        """
        # BCE Loss
        real_labels = torch.ones_like(real_outputs)
        fake_labels = torch.zeros_like(fake_outputs)
        real_loss = self.bce_logits(real_outputs, real_labels)
        fake_loss = self.bce_logits(fake_outputs, fake_labels)
        bce_loss = (real_loss + fake_loss) / 2

        # Gradient Penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=real_images.device)
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return bce_loss + gradient_penalty
    
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, discriminator_pred: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(discriminator_pred)
        return self.bce_logits(discriminator_pred, real_labels)
