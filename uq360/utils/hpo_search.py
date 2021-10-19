
import warnings
import sys
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from tqdm import tqdm


class CustomRandomSearch(RandomizedSearchCV):
    _required_parameters = ["estimator", "param_distributions"]

    def __init__(self, estimator, param_distributions, progress_bar=True, callback=None, n_iter=10, scoring=None,
                 n_jobs=None, refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise', return_train_score=False, model_stage=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.callback = callback
        self.model_stage = model_stage
        super().__init__(
            estimator=estimator, param_distributions=self.param_distributions,
            n_iter=self.n_iter, random_state=self.random_state,
            scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        params = list(ParameterSampler(self.param_distributions,
                                       self.n_iter, random_state=self.random_state))
        for idx, param in enumerate(tqdm(params, desc="Optimising Drift Detection Model...", file=sys.stdout,
                                         unit="models", dynamic_ncols=True, disable=not(self.progress_bar))):
            evaluate_candidates([param])
            if self.callback:
                stop = self.callback(self.model_stage, progress_step=idx+1,
                                     progress_total_count=len(params))  # idx starts at 0
                if stop:
                    break
