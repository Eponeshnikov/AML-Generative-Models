import torch.nn as nn
import torch
import torch.nn.functional as F


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
