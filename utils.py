import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class UNSWDataset(Dataset):

    def __init__(self, X, y, device=torch.device('cpu')):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.torch.from_numpy(X).to(torch.float32).to(device)
            self.y = torch.from_numpy(y).to(torch.int32).to(device)
        else:
            self.X = X.to(torch.float32).to(device)
            self.y = y.to(torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_reconstruction_loss(n_features, loss):
    def reconstruction_loss(input_and_mask, y_pred):
        X_values = input_and_mask[:, :n_features]
        missing_mask = input_and_mask[:, n_features:]
        observed_mask = 1 - missing_mask
        X_values_observed = X_values * observed_mask
        pred_observed = y_pred * observed_mask
        return loss(X_values_observed, pred_observed)

    return reconstruction_loss


def regularized_loss(X, output, criterion, model, l):
    criterion_loss = criterion(X, output)
    loss = 0
    values = X
    model_children = list(model.children())
    for i in range(len(model_children)):
        values = F.relu((model_children[i](values)))
        loss += torch.mean(torch.abs(values))
    return l * loss + criterion_loss
