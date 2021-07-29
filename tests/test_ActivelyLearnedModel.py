import unittest

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


class TestActivelyLearnedModel(unittest.TestCase):

    def test_fit_predict_and_metrics(self):
        from uq360.algorithms.actively_learned_model import ActivelyLearnedModel
        from uq360.algorithms.ensemble_heteroscedastic_regression import EnsembleHeteroscedasticRegression
        from uq360.metrics import compute_regression_metrics
        import torch
        import numpy as np

        def sample_(n):
            return np.random.rand(n,3)
        def querry_(x):
            return (np.cos(x[:,0]*np.log(x[:,1])+x[:,0]**2*x[:,2]) + np.random.rand(x.shape[0])).reshape(-1,1) 

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define config for Heteroscedastic regression
        config_HR = {"num_features": 3, "num_outputs": 1, "num_hidden": 50, "batch_size": 32, "num_epochs": 50,
                        "lr": 0.001}
        HR_kwargs = {"model_type":'mlp',
                    "config": config_HR,
                    "device": device}
        # define config for ensemble
        config_ensemble = {"num_models": 5, 
                "batch_size": 32,
                "model_kwargs":HR_kwargs, }

        # define config for active learning object
        config_AL = {"num_init": 100, 
        "T": 3, 
        "K": 100, 
        "M": 4, 
        "sampling_function": sample_, 
        "querry_function" : querry_,
        "model_function": EnsembleHeteroscedasticRegression,
        "model_kwargs": {"model_type":'ensembleheteroscedasticregression', 
                                                    "config":config_ensemble, 
                                                    "device":device}, }


        uq_model = ActivelyLearnedModel(config=config_AL, device=device).fit()

        X = sample_(1000)
        y = querry_(X)
        yhat, yhat_lb, yhat_ub = uq_model.predict(X)

        results = compute_regression_metrics(y, yhat, yhat_lb, yhat_ub)

        coverage = results["picp"]
        avg_width = results["mpiw"]
        rmse = results["rmse"]
        nll = results["nll"]
        auucc_gain = results["auucc_gain"]

        assert (coverage > 0.0)


if __name__ == '__main__':
    unittest.main()
