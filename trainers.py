from utils import make_reconstruction_loss, regularized_loss
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class AETrainer:
    def __init__(self, data, model, regularization=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data)
        self.model = model(input_shape=self.data.shape[1] * 2).to(self.device)
        self.regularization = regularization

    def fill(self, missing_mask):
        self.data[missing_mask] = 1

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

    def _create_dataloader(self, batch_size):
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        input_with_mask = torch.hstack([self.data, ~missing_mask])
        self.X_train, self.x_test = train_test_split(input_with_mask, test_size=0.2, random_state=42)
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