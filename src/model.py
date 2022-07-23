import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class CVAE(nn.Module):
    def __init__(self, latent_size=32, num_class=10):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.num_class = num_class

        #Encoder
        self.conv1 = nn.Conv2d(2,16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(4*4*32,300)
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        #Decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300, 4*4*32)
        self.conv3 = nn.ConvTransponse2d(32, 16, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTransponse2d(16, 1, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTransponse2d(1, 1, kernel_size=4)
    
    def encoder(self, x, c):
        y = torch.argmax(y, dim=1).reshape(y.shape[0],1,1,1)
        y = torch.ones(x.shape).to(device)*y
        t = torch.cat((x,y), dim=1)

        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.reshape(x.shape[0], -1)

        t = F.relu(self.linear1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)

        return eps*std +mu

    def unFlatten(self,x):
        return x.reshape(x.shape[0], 32, 4, 4)

    def decoder(self, z):
        t = F.relu(self.linear2(z))
        t = F.relu(self.linear3(t))
        t = self.unFlatten(t)
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))

        return t
    
    
    def forward(self, x, y):
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)

        #Addition of Class condition
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar


