from tqdm import tqdm
import torch
from utils import lossFunction, oneHotEncoding, save_reconstructed_images
import numpy as np

batch_size=100

#model, dataloader, dataset, device, optimizer, criterion
def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    total_loss = 0

    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeroes((x.shape[0], 10))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()
            pred, mu, logvar = model(x, label)

            loss = lossFunction(x, pred, mu, logvar)
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]

        except Exception as e:
            print("Something went wrong in Train Function")
        
        total_loss /= len(train_loader.dataset) 

    return total_loss

def test(epoch, model, test_loader, optim):
    reconstruction_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeroes((x.shape[0], 10))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x, label)

                loss = lossFunction(x, pred, mu, logvar)
                loss.backward()
                optim.step()

                total_loss += loss.cpu().data.numpy()*x.shape[0]

            except Exception as e:
                print("Something went wrong in Test Function")
            
            total_loss /= len(test_loader.dataset) 

    return total_loss

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