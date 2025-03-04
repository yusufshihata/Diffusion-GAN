import unittest
import torch
from src.loss import GeneratorLoss, DiscriminatorLoss
from src.model import Generator


class TestGeneratorLoss(unittest.TestCase):
    def setUp(self):
        self.loss = GeneratorLoss()

    def test_generator_loss_output(self):
        z = Generator(100, (3, 28, 28))(torch.randn(1, 100))
        output = self.loss(z)

        self.assertEqual(output.shape, torch.Size([]))


class TestDiscriminatorLoss(unittest.TestCase):
    def setUp(self):
        self.loss = DiscriminatorLoss()
        self.batch_size = 32

    def test_discriminator_loss_output(self):
        x = Generator(100, (3, 28, 28))(
            torch.randn(1, 100)
        )  # Use the generator here to get normalized data so we can calc loss
        z = Generator(100, (3, 28, 28))(torch.randn(1, 100))
        output = self.loss(x, z)

        self.assertEqual(output.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
