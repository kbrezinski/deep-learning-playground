import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

tensor_transform = transforms.ToTensor()
 
dataset = datasets.MNIST(root="./data",train=True,
            download=True, transform=tensor_transform)

# (batch_size, channels, height, width) = (64, 1, 28, 28)
loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=128, shuffle = True)

class Autoencoder(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 12),
        )
        # latent dim of 8 neurons
        self.z_mean = torch.nn.Linear(12, 8)
        self.z_log_var = torch.nn.Linear(12, 8)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28*28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        latent_space = z_mean + torch.exp(z_log_var / 2.) * torch.randn_like(z_mean)
        x = self.decoder(latent_space)
        return x, z_mean, z_log_var

model = Autoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def kl_divergence(z_mean, z_log_var, axis=1):
    return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), axis=axis)

num_epochs = 10
for epoch in range(num_epochs):
    for data in loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        recon, z_mean, z_log_var = model(img)
        loss = criterion(recon, img) + kl_divergence(z_mean, z_log_var, axis=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")