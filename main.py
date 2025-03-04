import argparse
import yaml
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from utils.loss import GeneratorLoss, DiscriminatorLoss
from scripts.train import train

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Usage: --mode [train, inference]")
    parser.add_argument("--dataset", type=str, required=True, help="Usage: --dataset Dataset to do training on [cifar, mnist].")
    
    args, _ = parser.parse_known_args()
    
    if args.mode == "train":
        parser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train.")
        parser.add_argument("--optimizer", type=str, required=False, help="Optimizer: [SGD, Adam].")
        parser.add_argument("--glr", type=float, required=False, help="Generator Learning Rate.")
        parser.add_argument("--dlr", type=float, required=False, help="Discriminator Learning Rate.")
        args = parser.parse_args()

        config = load_config(f"configs/{args.dataset}.yaml")

        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.optimizer is not None:
            config["optimizer"]["type"] = args.optimizer
        if args.glr is not None:
            config["optimizer"]["generator"]["lr"] = args.glr
        if args.dlr is not None:
            config["optimizer"]["discriminator"]["lr"] = args.dlr

        device = "cuda" if torch.cuda.is_available() else "cpu"

        gen = Generator(config["training"]["latent_dim"], config["training"]["img_channels"]).to(device)
        disc = Discriminator(config["training"]["img_channels"]).to(device)

        gcriterion = GeneratorLoss().to(device)
        dcriterion = DiscriminatorLoss().to(device)

        optimizer_type = config["optimizer"]["type"].lower()
        if optimizer_type == "sgd":
            Goptimizer = optim.SGD(gen.parameters(), lr=config["optimizer"]["generator"]["lr"])
            Doptimizer = optim.SGD(disc.parameters(), lr=config["optimizer"]["discriminator"]["lr"])
        elif optimizer_type == "adam":
            Goptimizer = optim.Adam(gen.parameters(),
                                    lr=config["optimizer"]["generator"]["lr"],
                                    betas=tuple(config["optimizer"]["generator"]["betas"]))
            Doptimizer = optim.Adam(disc.parameters(),
                                    lr=config["optimizer"]["discriminator"]["lr"],
                                    betas=tuple(config["optimizer"]["discriminator"]["betas"]))
        else:
            raise RuntimeError("Usage: --optimizer [SGD, Adam]")

        if args.dataset == "mnist":
            transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            trainset = MNIST(root="data/", train=True, transform=transforms, download=True)
        elif args.dataset == "cifar":
            transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            trainset = CIFAR10(root="data/", train=True, transform=transforms, download=True)
        else:
            raise RuntimeError("Usage: --dataset [mnist, cifar]")

        trainloader = DataLoader(
            trainset, batch_size=config["training"]["batch_size"], shuffle=True
        )

        train(
            gen,
            disc,
            Goptimizer,
            Doptimizer,
            gcriterion,
            dcriterion,
            trainloader,
            config["training"]["epochs"],
            config["training"]["checkpoint_interval"],
        )

if __name__ == "__main__":
    main()
