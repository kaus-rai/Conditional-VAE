import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim) -> None:
        super(CVAE, self).__init__()

        #Encoder
        self.fc1 = nn.Linear(x_dim + c_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, z_dim)
        self.fc3 = nn.Linear(h_dim2, z_dim)
        self.fc4 = nn.Linear(h_dim2, z_dim)

        #Decoder
        self.fc5 = nn.Linear(z_dim + c_dim, h_dim2)
        self.fc6 = nn.Linear(h_dim2, h_dim1)
        self.fc7 = nn.Linear(h_dim1, x_dim)


    def encoder(self, x, c):
        concat_input = torch.cat([x,c], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))

        return self.fc3(h), self.fc4(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decoder(self, z, c):
        concat_input = torch.cat([z,c], 1)
        h = F.relu(self.fc5(concat_input))
        h = F.relu(self.fc6(h))
        return F.sigmoid(self.fc7(h))


    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, 784), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var


