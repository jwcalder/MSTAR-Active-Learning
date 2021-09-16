import torch.nn as nn
import torch.nn.functional as F
import torch

#CNN for classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        w = (32,64) #Number of channels in 1st and 2nd layers
        self.conv1 = nn.Conv2d(3, w[0], 3, 1)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        f = 512  #Number of hidden nodes in fully connected layers
        self.fc1 = nn.Linear(w[1]*10*10, f)
        self.fc2 = nn.Linear(f, 10)
        self.bn1 = nn.BatchNorm1d(f)

    def forward(self, x): #88
        x = self.conv1(x) #86
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43
        x = self.conv2(x) #41
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)   #batch normalization
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    #This is useful for extracting features from convolutional part of NN
    def encode(self, x):
        x = self.conv1(x) #86
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43
        x = self.conv2(x) #41
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

#CNN Variational Autoencoder
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
 
        kernel_size = 3 # (3, 3) kernel
        num_classes = 10
        init_channels = 8 # initial number of filters
        image_channels = 3
        latent_dim = 32 # latent dimension for sampling

        # encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size, padding=1)
        self.enc2 = nn.Conv2d(init_channels, 2*init_channels, kernel_size, padding=1)
        self.enc3 = nn.Conv2d(2*init_channels, 4*init_channels, kernel_size, padding=1)
        self.enc4 = nn.Conv2d(4*init_channels, 64, kernel_size, stride=2, padding=1)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64*11*11, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(64, 8*init_channels, kernel_size, output_padding=1, dilation=3)
        self.dec2 = nn.ConvTranspose2d(8*init_channels, 4*init_channels, kernel_size, stride=2, output_padding=1, dilation=3)
        self.dec3 = nn.ConvTranspose2d(4*init_channels, 2*init_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(2*init_channels, image_channels, kernel_size, stride=2, padding=1, output_padding=1)

        #Dropout
        self.dropout1 = nn.Dropout(0.5)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x)) #88
        x = F.max_pool2d(x, 2) #44
        x = F.relu(self.enc2(x)) #44
        x = F.max_pool2d(x, 2) #22
        x = F.relu(self.enc3(x)) #22
        x = F.relu(self.enc4(x)) #11
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

    def decode(self, x):
 
        x = x.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(x)) #8
        x = F.relu(self.dec2(x)) #22
        x = self.dropout1(x)
        x = F.relu(self.dec3(x))
        x = torch.sigmoid(self.dec4(x))

        return x

    def forward(self, x): #88

        #encode
        x = self.encode(x)
        hidden = self.fc1(x) #(64*11*11, 128)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var) #(64, 16)
        z = self.fc2(z) #(64, 64)

        #decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var



