from tqdm import tqdm
import torch
from utils import lossFunction, oneHotEncoding, save_reconstructed_images

batch_size=100

def train(model, dataloader, dataset, device, optimizer, criterion):
    cond_dim = dataloader.dataset.train_labels.unique().size(0)
    model.train()
    running_loss = 0.0
    counter = 0
    for i, (data, cond) in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data, cond = data, oneHotEncoding(cond, cond_dim)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data, cond)
        # bce_loss = criterion(reconstruction, data)
        loss = lossFunction(reconstruction, data, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    cond_dim = dataloader.dataset.train_labels.unique().size(0)
    model.eval()
    running_loss = 0.0
    test_loss = []
    counter = 0
    with torch.no_grad():
        for i, (data, cond) in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data, cond = data, oneHotEncoding(cond, cond_dim)
            reconstruction, mu, logvar = model(data, cond)
            # bce_loss = criterion(reconstruction, data)
            running_loss += lossFunction(reconstruction, data, mu, logvar).item()

            if(i == int(len(dataset)/dataloader.batch_size)-1):
                recon_images = reconstruction
    test_loss = running_loss / counter 
    return test_loss,recon_images 