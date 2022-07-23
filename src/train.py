import model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from utils import save_reconstructed_images, save_loss_plot, generate_image, save_image
from engine import train, validate, test
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
learning_rate = 1e-3
epoches = 100
device = torch.device("cuda")
num_workers = 5
load_epoch = -1
generate = True


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

trainset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

model = model.CVAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

train_loss = []
test_loss = []

for epoch in range(epoches):
    print(f"Epoch {epoch+1} of {epoches}")
    model.train()
    train_epoch_loss = train(
        epoch, model, trainloader, optimizer
    )

    with torch.no_grad():
        test_epoch_loss, recon_images = test(
            epoch, model, testloader
        )

        if(generate):
            z = torch.randn(6,32).to(device)
            y = torch.tensor([1,2,3,4,5,6])-1

            generate_image(epoch, z, y, model)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)

    save_image(model, epoch)

    # save the reconstructed images from the validation loop
    # save_reconstructed_images(recon_images, epoch+1)
    # # convert the reconstructed images to PyTorch image grid format
    # image_grid = make_grid(recon_images.detach().cpu())
    # grid_images.append(image_grid)
    # print(f"Train Loss: {train_epoch_loss:.4f}")
    # print(f"Val Loss: {valid_epoch_loss:.4f}")

    # save the reconstructions as a .gif file
# image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, test_loss)
print('TRAINING COMPLETE')




