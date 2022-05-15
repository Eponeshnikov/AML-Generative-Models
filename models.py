import torch
import torch.nn as nn
import numpy as np


def gen_layers(in_, out_, layers, degree=2):
    neurons = [int(in_)]
    layer = in_
    for i in range(layers):
        layer = layer // degree
        neurons.append(int(layer))
    z = layer
    for i in range(layers - 1):
        layer = int(layer * (np.power((out_ / z), 1 / layers)))
        neurons.append(int(layer))
    neurons.append(int(out_))
    return neurons


class AE(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        in_dim = input_shape
        self.neurons = gen_layers(in_dim, in_dim / 2, 2, 4)
        self.encoder_layers = [nn.Linear(in_features=self.neurons[i], out_features=self.neurons[i + 1])
                               for i in range(len(self.neurons) // 2)]
        self.decoder_layers = [nn.Linear(in_features=self.neurons[i], out_features=self.neurons[i + 1])
                               for i in range(len(self.neurons) // 2, len(self.neurons) - 1)]

        self.enc_relu = []
        for en in self.encoder_layers:
            self.enc_relu.append(en)
            self.enc_relu.append(nn.ReLU())
        self.dec_relu = []
        for de in self.decoder_layers:
            self.dec_relu.append(de)
            self.dec_relu.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.enc_relu)
        self.decoder = nn.Sequential(*self.dec_relu)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# VAE model
class VAE(nn.Module):
    def __init__(self, input_shape, alpha=1):
        # Autoencoder only requires 1 dimensional argument since input and output-size is the same
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_shape, input_shape // 4), nn.ReLU(),
                                     nn.BatchNorm1d(input_shape // 4, momentum=0.7),
                                     nn.Linear(input_shape // 4, input_shape // 16), nn.ReLU(),
                                     nn.BatchNorm1d(input_shape // 16, momentum=0.7),
                                     nn.Linear(input_shape // 16, input_shape // 32), nn.LeakyReLU())
        self.hidden2mu = nn.Linear(input_shape // 32, input_shape // 32)
        self.hidden2log_var = nn.Linear(input_shape // 32, input_shape // 32)
        self.alpha = alpha
        self.decoder = nn.Sequential(nn.Linear(input_shape // 32, input_shape // 16), nn.ReLU(),
                                     nn.Linear(input_shape // 16, input_shape // 8), nn.ReLU(),
                                     nn.Linear(input_shape // 8, int(input_shape / 2)), nn.ReLU())

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return self.decoder(hidden)
