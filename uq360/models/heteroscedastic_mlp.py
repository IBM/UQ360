import torch
import torch.nn.functional as F
from uq360.models.noise_models.heteroscedastic_noise_models import GaussianNoise

class GaussianNoiseMLPNet(torch.nn.Module):

    def __init__(self, num_features, num_outputs, num_hidden):
        super(GaussianNoiseMLPNet, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_hidden)
        self.fc_mu = torch.nn.Linear(num_hidden, num_outputs)
        self.fc_log_var = torch.nn.Linear(num_hidden, num_outputs)
        self.noise_layer = GaussianNoise()

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def loss(self, y_true=None, mu_pred=None, log_var_pred=None):
        return self.noise_layer.loss(y_true, mu_pred, log_var_pred, reduce_mean=True)