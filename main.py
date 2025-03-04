import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10
from models.generator import Generator
from models.discriminator import Discriminator
from utils.loss import GeneratorLoss, DiscriminatorLoss
from scripts.train import train

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.save_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="Usage: --mode [train, inference]")
    parser.add_argument('--device', type=str, required=True, help="Usage: --device [cuda, cpu]")
    parser.add_argument('--dataset', type=str, required=True, help="Usage: --dataset Dataset to do training on [cifar, mnist].")
    args = parser.parse_args()
    if args.mode == 'train':
        parser.add_argument('--epochs', type=int, required=False, help="Usage: --epochs Number of epochs to train the model for.")
        parser.add_argument('--optimizer', type=str, required=False, help="Usage: --optimizer [SGD, Adam].")
        parser.add_argument('--glr', type=float, required=False, help="Usage: --glr Generator Training Learning Rate.")
        parser.add_argument('--dlr', type=float, required=False, help="Usage: --dlr Discriminator Training Learning Rate.")

        args = parser.parse_args()

        config = load_config(f"configs/{args.dataset}.yaml")
        config['training']['epochs'] = args.epochs
        config['optimizer']['type'] = args.optimizer
        config['optimizer']['generator']['lr'] = args.glr
        config['optimizer']['discriminator']['lr'] = args.dlr

        if config['optimizer']['type'] == 'sgd'.upper():
            Goptimizer = optim.SGD(lr=config['optimizer']['generator']['lr'])
            Doptimizer = optim.SGD(lr=config['optimizer']['discriminator']['lr'])
        elif config['optimizer']['type'] == 'adam'.upper():
            Goptimizer = optim.Adam(lr=config['optimizer']['generator']['lr'], betas=config['optimizer']['generator']['betas'])
            Doptimizer = optim.Adam(lr=config['optimizer']['discriminator']['lr'], betas=config['optimizer']['discriminator']['betas'])
        else:
            raise RuntimeError("Usage: --optimizer [SGD, Adam]")
        
        epochs = config['training']['epochs']

        gen = Generator(config['training']['latent_dim'], config['training']['img_channels']).to(args.device)
        disc = Discriminator(config['training']['img_channels']).to(args.device)

        gcriterion = GeneratorLoss().to(args.device)
        dcriterion = DiscriminatorLoss().to(args.device)

        if args.dataset == 'mnist':
            transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])

            trainset = MNIST(root='data/', train=True, transform=transforms)
        elif args.dataset == 'cifar':
            transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            trainset = CIFAR10(root='data/', train=True, transform=transforms)
        else:
            raise RuntimeError("Usage: --dataset [MNIST, CIFAR10]")
        
        trainloader = DataLoader(trainset, batch_size=config['training']['batch_size'], shuffle=True)


        train(gen, disc, Goptimizer, Doptimizer, gcriterion, dcriterion, trainloader, epochs, config['training']['checkpoint_interval'])
