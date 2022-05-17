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
        for i, en in enumerate(self.encoder_layers):
            self.enc_relu.append(en)
            if en != len(self.encoder_layers)-1:
                self.enc_relu.append(nn.ReLU())
        self.dec_relu = []
        for i, de in enumerate(self.decoder_layers):
            self.dec_relu.append(de)
            if de != len(self.decoder_layers) - 1:
                self.enc_relu.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.enc_relu, nn.Sigmoid())
        self.decoder = nn.Sequential(*self.dec_relu, nn.Sigmoid())

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


class GeneratorModel(nn.Module):
    def __init__(self, input_dim_, output_dim, classes, embedding_dim):
        super(GeneratorModel, self).__init__()
        input_dim = input_dim_ + embedding_dim
        self.label_embedding = nn.Embedding(classes, embedding_dim)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output


class DiscriminatorModel(nn.Module):
    def __init__(self, input_dim_, classes, embedding_dim, output_dim=1):
        super(DiscriminatorModel, self).__init__()
        input_dim = input_dim_ + embedding_dim
        self.label_embedding = nn.Embedding(classes, embedding_dim)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            #nn.BatchNorm1d(512, momentum=0.1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 128),
            #nn.BatchNorm1d(512, momentum=0.1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            #nn.BatchNorm1d(512, momentum=0.1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)

        return output