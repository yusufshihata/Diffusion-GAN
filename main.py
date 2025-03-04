import argparse
import yaml
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from utils.loss import GeneratorLoss, DiscriminatorLoss
from scripts.train import train
from scripts.inference import inference

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "inference"],
        help="Usage: --mode [train, inference]"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["cifar", "mnist"],
        help="Dataset to use for training/inference."
    )
    
    # Arguments for training mode
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train the model for.")
    parser.add_argument("--optimizer", type=str, required=False, choices=["SGD", "Adam"], help="Optimizer type.")
    parser.add_argument("--glr", type=float, required=False, help="Generator Learning Rate.")
    parser.add_argument("--dlr", type=float, required=False, help="Discriminator Learning Rate.")

    # Arguments for inference mode
    parser.add_argument("--latent_dim", type=int, required=False, help="Latent vector size for inference.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the trained generator checkpoint.")

    args = parser.parse_args()
    
    config = load_config(f"configs/{args.dataset}.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        config["training"]["epochs"] = args.epochs
        config["optimizer"]["type"] = args.optimizer
        config["optimizer"]["generator"]["lr"] = args.glr
        config["optimizer"]["discriminator"]["lr"] = args.dlr

        # Optimizer selection
        if config["optimizer"]["type"].lower() == "sgd":
            Goptimizer = optim.SGD(lr=config["optimizer"]["generator"]["lr"])
            Doptimizer = optim.SGD(lr=config["optimizer"]["discriminator"]["lr"])
        elif config["optimizer"]["type"].lower() == "adam":
            Goptimizer = optim.Adam(
                lr=config["optimizer"]["generator"]["lr"],
                betas=config["optimizer"]["generator"]["betas"],
            )
            Doptimizer = optim.Adam(
                lr=config["optimizer"]["discriminator"]["lr"],
                betas=config["optimizer"]["discriminator"]["betas"],
            )
        else:
            raise RuntimeError("Usage: --optimizer [SGD, Adam]")

        epochs = config["training"]["epochs"]

        gen = Generator(config["training"]["latent_dim"], config["training"]["img_channels"]).to(device)
        disc = Discriminator(config["training"]["img_channels"]).to(device)

        gcriterion = GeneratorLoss().to(device)
        dcriterion = DiscriminatorLoss().to(device)

        # Dataset selection
        if args.dataset.lower() == "mnist":
            transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            trainset = MNIST(root="data/", train=True, transform=transforms, download=True)
        elif args.dataset.lower() == "cifar":
            transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            trainset = CIFAR10(root="data/", train=True, transform=transforms, download=True)
        else:
            raise RuntimeError("Usage: --dataset [MNIST, CIFAR10]")

        trainloader = DataLoader(trainset, batch_size=config["training"]["batch_size"], shuffle=True)

        train(gen, disc, Goptimizer, Doptimizer, gcriterion, dcriterion, trainloader, epochs, config["training"]["checkpoint_interval"])

    elif args.mode == "inference":
        if not args.checkpoint:
            raise ValueError("Inference mode requires --checkpoint argument to load the trained model.")
        if not args.latent_dim:
            args.latent_dim = config["training"]["latent_dim"]  # Use default latent dimension from config

        # Load generator
        gen = Generator(args.latent_dim, config["training"]["img_channels"]).to(device)
        gen.load_state_dict(torch.load(args.checkpoint, map_location=device))
        gen.eval()

        # Generate images
        inference(gen, args.num_samples)

if __name__ == "__main__":
    main()
