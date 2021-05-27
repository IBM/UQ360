from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from uq360.algorithms.builtinuq import BuiltinUQ

np.random.seed(42)
torch.manual_seed(42)


class _MLPNet_Main(torch.nn.Module):
    def __init__(self, num_features, num_outputs, num_hidden):
        super(_MLPNet_Main, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_hidden)
        self.fc_mu = torch.nn.Linear(num_hidden, num_outputs)
        self.fc_log_var = torch.nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class _MLPNet_Aux(torch.nn.Module):
    def __init__(self, num_features, num_outputs, num_hidden):
        super(_MLPNet_Aux, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_hidden)
        self.fc_log_var = torch.nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc(x))
        log_var = self.fc_log_var(x)
        return log_var


class AuxiliaryIntervalPredictor(BuiltinUQ):
    """ Auxiliary Interval Predictor [1]_ uses an auxiliary model to encourage calibration of the main model.

    References:
        .. [1] Thiagarajan, J. J., Venkatesh, B., Sattigeri, P., & Bremer, P. T. (2020, April). Building calibrated deep
         models via uncertainty matching with auxiliary interval predictors. In Proceedings of the AAAI Conference on
         Artificial Intelligence (Vol. 34, No. 04, pp. 6005-6012). https://arxiv.org/abs/1909.04079
    """

    def __init__(self, model_type=None, main_model=None, aux_model=None, config=None, device=None, verbose=True):
        """
        Args:
            model_type: The model type used to build the main model and the auxiliary model. Currently supported values
             are [mlp, custom]. `mlp` modeltype learns a mlp neural network using pytorch framework. For `custom` the user
             provide `main_model` and `aux_model`.
            main_model: (optional) The main prediction model. Currently support pytorch models that return mean and log variance.
            aux_model: (optional) The auxiliary prediction model. Currently support pytorch models that return calibrated log variance.
            config: dictionary containing the config parameters for the model.
            device: device used for pytorch models ignored otherwise.
            verbose: if True, print statements with the progress are enabled.
        """

        super(AuxiliaryIntervalPredictor).__init__()
        self.config = config
        self.device = device
        self.verbose = verbose
        if model_type == "mlp":
            self.model_type = model_type
            self.main_model = _MLPNet_Main(
                num_features=self.config["num_features"],
                num_outputs=self.config["num_outputs"],
                num_hidden=self.config["num_hidden"],
            )
            self.aux_model = _MLPNet_Aux(
                num_features=self.config["num_features"],
                num_outputs=self.config["num_outputs"],
                num_hidden=self.config["num_hidden"],
            )
        elif model_type == "custom":
            self.model_type = model_type
            self.main_model = main_model
            self.aux_model = aux_model

        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        return {"model_type": self.model_type, "config": self.config, "main_model": self.main_model,
                "aux_model": self.aux_model, "device": self.device, "verbose": self.verbose}

    def _main_model_loss(self, y_true, y_pred_mu, y_pred_log_var, y_pred_log_var_aux):
        r = torch.abs(y_true - y_pred_mu)
        # + 0.5 * y_pred_log_var +
        loss = torch.mean(0.5 * torch.exp(-y_pred_log_var) * r ** 2) + \
               self.config["lambda_match"] * torch.mean(torch.abs(torch.exp(0.5 * y_pred_log_var) - torch.exp(0.5 * y_pred_log_var_aux)))
        return loss

    def _aux_model_loss(self, y_true, y_pred_mu, y_pred_log_var_aux):
        deltal = deltau = 2.0 * torch.exp(0.5 * y_pred_log_var_aux)
        upper = y_pred_mu + deltau
        lower = y_pred_mu - deltal
        width = upper - lower
        r = torch.abs(y_true - y_pred_mu)

        emce = torch.mean(torch.sigmoid((y_true - lower) * (upper - y_true) * 100000))

        loss_emce = torch.abs(self.config["calibration_alpha"]-emce)
        loss_noise = torch.mean(torch.abs(0.5 * width - r))
        loss_sharpness = torch.mean(torch.abs(upper - y_true)) + torch.mean(torch.abs(lower - y_true))

        #print(emce)
        return loss_emce + self.config["lambda_noise"] * loss_noise + self.config["lambda_sharpness"] * loss_sharpness

    def fit(self, X, y):
        """ Fit the Auxiliary Interval Predictor model.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """

        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        dataset_loader = DataLoader(
            TensorDataset(X,y),
            batch_size=self.config["batch_size"]
        )
        optimizer_main_model = torch.optim.Adam(self.main_model.parameters(), lr=self.config["lr"])
        optimizer_aux_model = torch.optim.Adam(self.aux_model.parameters(), lr=self.config["lr"])

        for it in range(self.config["num_outer_iters"]):

            # Train the main model
            for epoch in range(self.config["num_main_iters"]):
                avg_mean_model_loss = 0.0
                for batch_x, batch_y in dataset_loader:
                    self.main_model.train()
                    self.aux_model.eval()
                    batch_y_pred_log_var_aux = self.aux_model(batch_x)
                    batch_y_pred_mu, batch_y_pred_log_var = self.main_model(batch_x)
                    main_loss = self._main_model_loss(batch_y, batch_y_pred_mu, batch_y_pred_log_var, batch_y_pred_log_var_aux)
                    optimizer_main_model.zero_grad()
                    main_loss.backward()
                    optimizer_main_model.step()

                    avg_mean_model_loss += main_loss.item()/len(dataset_loader)

                if self.verbose:
                    print("Iter: {},  Epoch: {}, main_model_loss = {}".format(it, epoch, avg_mean_model_loss))

            # Train the auxiliary model
            for epoch in range(self.config["num_aux_iters"]):
                avg_aux_model_loss = 0.0
                for batch_x, batch_y in dataset_loader:
                    self.aux_model.train()
                    self.main_model.eval()
                    batch_y_pred_log_var_aux = self.aux_model(batch_x)
                    batch_y_pred_mu, batch_y_pred_log_var = self.main_model(batch_x)
                    aux_loss = self._aux_model_loss(batch_y, batch_y_pred_mu, batch_y_pred_log_var_aux)
                    optimizer_aux_model.zero_grad()
                    aux_loss.backward()
                    optimizer_aux_model.step()

                    avg_aux_model_loss += aux_loss.item() / len(dataset_loader)

                if self.verbose:
                    print("Iter: {},  Epoch: {}, aux_model_loss = {}".format(it, epoch, avg_aux_model_loss))

        return self

    def predict(self, X, return_dists=False):
        """
        Obtain predictions for the test points.

        In addition to the mean and lower/upper bounds, also returns full predictive distribution (return_dists=True).

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            return_dists: If True, the predictive distribution for each instance using scipy distributions is returned.

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
            dists: list of predictive distribution as `scipy.stats` objects with length n_samples.
                Only returned when `return_dists` is True.
        """

        self.main_model.eval()

        X = torch.from_numpy(X).float().to(self.device)
        dataset_loader = DataLoader(
            X,
            batch_size=self.config["batch_size"]
        )

        y_mean_list = []
        y_log_var_list = []
        for batch_x in dataset_loader:
            batch_y_pred_mu, batch_y_pred_log_var = self.main_model(batch_x)
            y_mean_list.append(batch_y_pred_mu.data.cpu().numpy())
            y_log_var_list.append(batch_y_pred_log_var.data.cpu().numpy())

        y_mean = np.concatenate(y_mean_list)
        y_log_var = np.concatenate(y_log_var_list)
        y_std = np.sqrt(np.exp(y_log_var))
        y_lower = y_mean - 2.0*y_std
        y_upper = y_mean + 2.0*y_std

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        if return_dists:
            dists = [norm(loc=y_mean[i], scale=y_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_dists',))
            res = Result(*res, y_dists=dists)

        return res
