import numpy as np

import pandas as pd
from uq360.algorithms.builtinuq import BuiltinUQ



class ActivelyLearnedModel(BuiltinUQ):
    """ActivelyLearnedModel assumes an existing BuiltinUQ model, and implements an active learning training of this model. This code is supporting Pestourie et al. "Active learning of deep surrogates for PDEs: application to metasurface design." npj Computational Materials 6.1 (2020): 1-7."""

    def __init__(self, config=None, device=None, verbose=True, online=True):
        """Initializer for Actively learned model. 
        Args: 
            config: dictionary containing the config parameters for the model. For active learning: num_init, T, K, M, sampling_function, querry_function, for the used model:
                {"model_function": BuilInUQ model to actively learn,
                "model_args": same arguments as the BuilInUQ model used,
                "model_kwargs": same keyword arguments as the BuilInUQ model used}
            device: device used for pytorch models ignored otherwise.
            """
        super(ActivelyLearnedModel, self).__init__()
        self.config = config
        self.device = device
        self.verbose = verbose  
        self.online = online
        self.X = [] #will keep track of the training set
        self.y = [] #will keep track of the expensive querries
        self.builtinuqmodel = self.config["model_function"](*self.config.get("model_args", ()), **self.config.get("model_kwargs", {}))


    def _choose_highest_var_points(self, X, var, K):
        
        if len(var.shape)==2:
            pred_dict = {'indices': np.arange(len(var[:,0])),
                 'var': var[:,0]
                }
        else:
            pred_dict = {'indices': np.arange(len(var)),
                 'var': var
                }

        pred_df = pd.DataFrame(data=pred_dict)
        pred_df_sorted = pred_df.sort_values(by='var')

        return X[pred_df_sorted["indices"][-K:].values, :]
        
            
    def _choose_highest_var_points_and_labels(self, X, var, K, y):
        
        if len(var.shape)==2:
            pred_dict = {'indices': np.arange(len(var[:,0])),
                 'var': var[:,0]
                }
        else:
            pred_dict = {'indices': np.arange(len(var)),
                 'var': var
                }

        pred_df = pd.DataFrame(data=pred_dict)
        pred_df_sorted = pred_df.sort_values(by='var')


        return X[pred_df_sorted["indices"][-K:].values, :], y[pred_df_sorted["indices"][-K:].values, :] 


    def fit(self):
        """ Fit the actively learned model, by increasing the dataset efficiently. NB: it does not take a dataset as argument, because it is building one during training.
        Returns:
            self
        """

        if self.online:
            self.X = self.config["sampling_function"](self.config["num_init"])
            self.y = self.config["querry_function"](self.X)
        else :
            self.X = self.config["sampling_function"](0, self.config["num_init"])
            self.y = self.config["querry_function"](0, self.config["num_init"])
        self.builtinuqmodel.fit(self.X, self.y)

        for i in range(self.config["T"]):
            self.verbose and print(f"\nT = {i}\n")

            if self.online:
                X_proposed = self.config["sampling_function"](self.config["M"]*self.config["K"])
                res = self.builtinuqmodel.predict(X_proposed)
                X_added = self._choose_highest_var_points(X_proposed, res.y_upper-res.y_lower, self.config["K"]) #use y_upper-y_lower as a proxy of the variance (up to an increasing function)
                y_added = self.config["querry_function"](X_added)
            else:
                start_index = int(self.config["num_init"]+i*self.config["M"]*self.config["K"])
                n_points = int(self.config["M"]*self.config["K"])
                X_proposed = self.config["sampling_function"](start_index, n_points)
                y_proposed = self.config["querry_function"](start_index, n_points)
                res = self.builtinuqmodel.predict(X_proposed)
                X_added, y_added = self._choose_highest_var_points_and_labels(X_proposed, res.y_upper-res.y_lower, self.config["K"], y_proposed)

            self.X = np.vstack((self.X,X_added))

            if len(self.y.shape)==2:
                self.y = np.vstack((self.y,y_added)) #2d array
            else:
                self.y = np.hstack((self.y,y_added)) #1d array
            
            self.builtinuqmodel.fit(self.X, self.y)

        return self

    def predict(self, X):
        """
        Obtain predictions for the test points.
        In addition to the mean and lower/upper bounds, also returns epistemic uncertainty (return_epistemic=True)
        and full predictive distribution (return_dists=True).
        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
        Returns:
            namedtuple: A namedtupe that holds
            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
        """
        return self.builtinuqmodel.predict(X)
