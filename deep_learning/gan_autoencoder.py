
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="data/", train=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
            
class GAN(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 28*28),
            torch.nn.Tanh()
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(28*28, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(128, 1),
        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GAN().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
opt_disc = torch.optim.Adam(model.generator.parameters(), lr=1e-5)
opt_gen = torch.optim.Adam(model.generator.parameters(), lr=1e-3)

def train_gan(loader, epochs=100):

    for epoch in range(epochs):
        
        model.train()
        losses = {'disc_fake': 0, 'disc_real': 0, 'gen': 0}
        for features, _ in loader:

            batch_size = features.size(0)
            opt_disc.zero_grad()

            # Fetch real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size, device=device)

            # Generate fake images
            noise = torch.randn(batch_size, 64).to(device)
            fake_images = model.generator(noise)
            fake_labels = torch.zeros(batch_size, device=device)

            # Discriminator loss on real images
            real_outputs = model.discriminator(real_images).squeeze(-1)
            real_loss = criterion(real_outputs, real_labels)

            # Discriminator loss on fake images
            fake_outputs = model.discriminator(fake_images).squeeze(-1)
            fake_loss = criterion(fake_outputs, fake_labels)

            # Combined loss and backprop
            disc_loss = .5 * (real_loss + fake_loss)
            disc_loss.backward()
            opt_disc.step()

            # Train generator
            opt_gen.zero_grad()
            flipped_labels = real_labels
            # Detach the fake images from the discriminator; avoids backprop to generator
            gen_fake_outputs = model.discriminator(fake_images.detach()).squeeze(-1)
            gen_loss = criterion(gen_fake_outputs, flipped_labels)
            gen_loss.backward()
            opt_gen.step()

            losses['disc_fake'] += fake_loss.item()
            losses['disc_real'] += real_loss.item()
            losses['gen'] += gen_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, disc_fake: {losses['disc_fake'] / len(loader):.4f}, disc_real: {losses['disc_real'] / len(loader):.4f}, gen: {losses['gen'] / len(loader):.4f}")


train_gan(loader, epochs=10)

import matplotlib.pyplot as plt

with torch.no_grad():
    model.eval()
    noise = torch.randn(16, 64).to(device)
    fake_images = model.generator(noise)
    fake_images = fake_images.view(-1, 1, 28, 28)
    fake_images = fake_images.cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i, 0], cmap="gray")
        ax.axis("off")
    plt.show()

