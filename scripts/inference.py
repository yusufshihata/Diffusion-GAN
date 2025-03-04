import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def inference(
    generator_output: torch.Tensor,
    num_images: int = 16,
    title: str = "Generated Images",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize and optionally save a batch of images from the generator output.

    Args:
        generator_output (torch.Tensor): Output from the generator [batch_size, 1, 28, 28]
        num_images (int): Number of images to display (default: 16)
        title (str): Title for the plot
        save_path (str, optional): Path to save the image (e.g., "output.png"), None if no save
    """
    batch_size = generator_output.size(0)
    num_images = min(num_images, batch_size)

    images = generator_output[:num_images].detach().cpu()

    images = (images + 1) / 2

    images = images.squeeze(1).numpy()

    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=16)

    axes = axes.flatten() if num_images > 1 else [axes]

    for i in range(num_images):
        axes[i].imshow(images[i], cmap="gray")
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

    plt.show()
