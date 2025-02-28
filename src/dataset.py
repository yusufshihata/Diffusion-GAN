from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class MNISTDigits(Dataset):
    def __init__(self):
        self.dataset = MNIST(root='data', train=True, download=True)
        self.data = self.dataset['data']
        self.targets = self.dataset['targets']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
