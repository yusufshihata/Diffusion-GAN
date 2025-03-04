import unittest
from src.model import Generator, Discriminator
import torch


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.img_shape = (3, 28, 28)
        self.out_shape = (1, 3 * 28 * 28)

    def test_generator_output(self):
        z = torch.randn(1, self.latent_dim)
        generator = Generator(self.latent_dim, self.img_shape)
        output = generator(z)

        self.assertEqual(output.shape, torch.Size(self.out_shape))


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.img_shape = (3, 28, 28)
        self.in_shape = (1, 3 * 28 * 28)

    def test_discriminator_output(self):
        x = torch.randn(1, self.in_shape[1])
        discriminator = Discriminator(self.img_shape)
        output = discriminator(x)

        self.assertEqual(output.shape, torch.Size([1, 1]))

    def test_probability_output(self):
        x = torch.randn(1, self.in_shape[1])
        discriminator = Discriminator(self.img_shape)
        output = discriminator(x)

        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


if __name__ == "__main__":
    unittest.main()
