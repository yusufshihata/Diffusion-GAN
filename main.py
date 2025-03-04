import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.generator import Generator
from models.discriminator import Discriminator

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.save_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Usage: --dataset Dataset to do training on.")
    parser.add_argument('--epochs', type=int, required=True, help="Usage: --epochs Number of epochs to train the model for.")
    parser.add_argument('--optimizer', type=str, required=True, help="Usage: --optimizer Training Optimizer.")
    parser.add_argument('--glr', type=float, required=True, help="Usage: --glr Generator Training Learning Rate.")
    parser.add_argument('--dlr', type=float, required=True, help="Usage: --dlr Discriminator Training Learning Rate.")
    args = parser.parse_args()

    config = load_config(f"configs/{args.dataset}.yaml")
    config['training']['epochs'] = args.epochs