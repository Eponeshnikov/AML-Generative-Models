import numpy as np

from utils import make_reconstruction_loss, regularized_loss, UNSWDataset
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.model_selection import train_test_split


class AETrainer:
    def __init__(self, data_, model, regularization=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data_[0])
        self.labels = data_[1]
        self.model = model(input_shape=self.data.shape[1] * 2).to(self.device)
        self.regularization = regularization

    def fill(self, missing_mask):
        self.data[missing_mask] = 0

    def _create_missing_mask(self):
        return torch.isnan(self.data)

    def _train_epoch(self, model, optimizer, criterion):
        epoch_loss = 0
        model.train()

        for batch in self.train_data_loader:
            input_ = batch
            input_ = input_.to(self.device)
            optimizer.zero_grad()
            pred = model(input_)
            if not self.regularization:
                loss = criterion(input_, pred)
                loss1 = loss
            else:
                loss = regularized_loss(input_, pred, criterion, self.model, self.l1)
                loss1 = criterion(input_, pred)
            loss.backward()
            optimizer.step()
            epoch_loss += loss1.item()

        return epoch_loss / len(self.train_data_loader)  # , epoch_acc / len(data_loader)

    def test_epoch(self, model, criterion):
        model.eval()
        with torch.no_grad():
            pred = model(self.test_data_loader.dataset)
            loss = criterion(self.test_data_loader.dataset, pred)
        return loss.item()

    def fill_data(self):
        with torch.no_grad():
            train = self.model(self.train_data_loader.dataset)
            test = self.model(self.test_data_loader.dataset)
        return (train, self.y_train), (test, self.y_test)

    def _create_dataloader(self, batch_size):
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        input_with_mask = torch.hstack([self.data, ~missing_mask])
        self.X_train, self.x_test, self.y_train, self.y_test = train_test_split(input_with_mask, self.labels,
                                                                                test_size=0.2, random_state=42)
        self.train_data_loader = torch.utils.data.DataLoader(self.X_train.to(self.device), batch_size=batch_size,
                                                             shuffle=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.x_test.to(self.device), batch_size=batch_size,
                                                            shuffle=False)

    def train(self, train_epochs=100, batch_size=256, verbose_time=10, lr=1e-6, l1=0.3, loss=nn.MSELoss()):
        self.l1 = l1
        self._create_dataloader(batch_size)
        criterion = make_reconstruction_loss(self.data.shape[1], loss)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.history = {'train_loss': [], 'test_loss': []}
        for epoch in range(train_epochs):
            train_loss = self._train_epoch(self.model, optimizer, criterion)
            test_loss = self.test_epoch(self.model, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            # observed_mae = masked_mae(X_true=self.data, X_pred=X_pred, mask=observed_mask)
            if epoch % verbose_time == 0:
                print('epoch', epoch, "train_loss:", train_loss, 'test_loss:', test_loss)


class VAETrainer(AETrainer):
    def __init__(self, data, model):
        super(VAETrainer, self).__init__(data, model)

    def _train_epoch(self, model, optimizer, criterion):
        epoch_loss = 0
        model.train()

        for batch in self.train_data_loader:
            input_ = batch
            input_ = input_.to(self.device)
            optimizer.zero_grad()
            mu, log_var = model.encode(input_)
            x_reconst = model(input_)
            kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
            loss = criterion(input_, x_reconst) + kl_loss
            loss1 = criterion(input_, x_reconst)
            loss.backward()
            optimizer.step()
            epoch_loss += loss1.item()

        return epoch_loss / len(self.train_data_loader)

    def test_epoch(self, model, criterion):
        model.eval()
        with torch.no_grad():
            pred = model(self.test_data_loader.dataset)
            nans = pred.isnan()
            pred[nans] = 0
            loss = criterion(self.test_data_loader.dataset, pred)
        return loss.item()

    def fill_data(self):
        with torch.no_grad():
            train = self.model(self.train_data_loader.dataset)
            test = self.model(self.test_data_loader.dataset)
            train_nans = train.isnan()
            train[train_nans] = 0
            test_nans = test.isnan()
            test[test_nans] = 0
        return (train, self.y_train), (test, self.y_test)


class CGanTrainer:
    def __init__(self, data, generator, discriminator):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.hist_raw = np.histogram(self.data[0], bins=120)
        self.generator = generator
        self.discriminator = discriminator

    def create_dataloader(self, batch_size):
        self.dataset = UNSWDataset(self.data[0], self.data[1], device=self.device)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def init_model_params(self, input_dim_g=100, embedding_dim=100):
        self.input_dim_g = input_dim_g
        self.output_dim_g = self.dataset.X.shape[1]
        self.classes = torch.unique(self.dataset.y).shape[0]
        self.input_dim_d = self.output_dim_g
        self.embedding_dim = embedding_dim

    def init_models(self):
        self.discriminator = self.discriminator(input_dim_=self.input_dim_d, classes=self.classes,
                                                embedding_dim=self.embedding_dim).to(self.device)
        self.generator = self.generator(input_dim_=self.input_dim_g, output_dim=self.output_dim_g, classes=self.classes,
                                        embedding_dim=self.embedding_dim).to(self.device)

    def _train_epoch(self, generator_optimizer, discriminator_optimizer, loss, gen_updates=1, add_noise=False):
        G_loss = []
        D_loss = []
        for batch_idx, data_input in enumerate(self.data_loader):
            digit_labels = data_input[1]  # batch_size
            size = digit_labels.shape[0]

            noise = torch.randn(size, self.input_dim_g).to(self.device)
            fake_labels = torch.randint(0, self.classes, (size,)).to(self.device)
            generated_data = self.generator(noise, fake_labels)

            # Discriminator
            true_data = data_input[0]

            true_labels = torch.ones(size).to(self.device)

            discriminator_optimizer.zero_grad()
            if add_noise and len(self.history['loss_g']) > 1:
                noise_ = (1 + self.history['loss_g'][-1]) * torch.randn(true_data.shape).to(self.device)
                true_data += noise_
            discriminator_output_for_true_data = self.discriminator(true_data, digit_labels).view(size)
            true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
            if add_noise and len(self.history['loss_g']) > 1:
                noise_ = (1 + self.history['loss_g'][-1]) * torch.randn(generated_data.detach().shape).to(self.device)
                discriminator_output_for_generated_data = self.discriminator(generated_data.detach() + noise_,
                                                                             fake_labels).view(
                    size)
            else:
                discriminator_output_for_generated_data = self.discriminator(generated_data.detach(), fake_labels).view(
                    size)
            generator_discriminator_loss = loss(
                discriminator_output_for_generated_data, torch.zeros(size).to(self.device)
            )
            discriminator_loss = (
                                         true_discriminator_loss + generator_discriminator_loss
                                 ) / 2

            discriminator_loss.backward()
            discriminator_optimizer.step()

            D_loss.append(discriminator_loss.data.item())

            # Generator
            for i in range(gen_updates):
                noise = torch.randn(size, self.input_dim_g).to(self.device)
                generator_optimizer.zero_grad()
                # It's a choice to generate the data again
                generated_data = self.generator(noise, fake_labels)  # batch_size X 784
                discriminator_output_on_generated_data = self.discriminator(generated_data, fake_labels).view(size)
                generator_loss = loss(discriminator_output_on_generated_data, true_labels)
                generator_loss.backward()
                generator_optimizer.step()

            G_loss.append(generator_loss.data.item())

            return G_loss, D_loss

    def generate_data(self, label, size):
        with torch.no_grad():
            noise = torch.randn(size, self.input_dim_g).to(self.device)
            fake_labels = label * torch.ones(size).to(torch.int).to(self.device)
            generated_data = self.generator(noise, fake_labels).cpu()
        return generated_data

    def calc_kl(self):
        gen_data = self.generate_data(0, 100)
        for i in range(1, self.classes):
            gen_data = np.vstack([gen_data, self.generate_data(i, 100)])
        hist_gen = np.histogram(gen_data, bins=120)
        return entropy(hist_gen[0], self.hist_raw[0])

    def train(self, train_epoch=100, batch_size=256, input_dim_g=100, embedding_dim=100, lr=0.0005, verbose_time=1,
              gen_updates=1, add_noise=True):
        self.create_dataloader(batch_size)
        self.init_model_params(input_dim_g=input_dim_g, embedding_dim=embedding_dim)
        self.init_models()
        if isinstance(lr, float):
            lr1 = lr
            lr2 = lr
        elif isinstance(lr, list):
            lr1 = lr[0]
            lr2 = lr[1]
        else:
            print('lr=0.0005')
            lr1 = lr2 = 0.0005

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr1)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr2)
        loss = nn.BCELoss()
        self.history = {'loss_g': [], 'loss_d': [], 'kl': []}
        for epoch_idx in range(train_epoch):
            g_loss, d_loss = self._train_epoch(generator_optimizer, discriminator_optimizer, loss, gen_updates,
                                               add_noise=add_noise)
            self.history['loss_g'].append(torch.mean(torch.FloatTensor(g_loss)))
            self.history['loss_d'].append(torch.mean(torch.FloatTensor(d_loss)))
            kl = self.calc_kl()
            self.history['kl'].append(kl)
            if epoch_idx % verbose_time == 0:
                print('[%d/%d]: loss_d: %.3f, loss_g: %.3f, kl: %.3f' % (epoch_idx, train_epoch,
                                                               torch.mean(torch.FloatTensor(d_loss)),
                                                               torch.mean(torch.FloatTensor(g_loss)), kl))
