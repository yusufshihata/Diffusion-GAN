import torch
import os


def save_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    Goptimizer: torch.optim.Optimizer,
    Doptimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str = "models",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "Goptimizer_state_dict": Goptimizer.state_dict(),
        "Doptimizer_state_dict": Doptimizer.state_dict(),
    }
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    Goptimizer: torch.optim.Optimizer,
    Doptimizer: torch.optim.Optimizer,
    checkpoint_path: str,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    Goptimizer.load_state_dict(checkpoint["Goptimizer_state_dict"])
    Doptimizer.load_state_dict(checkpoint["Doptimizer_state_dict"])

    print(
        f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {checkpoint['epoch']}."
    )

    return checkpoint["epoch"]
