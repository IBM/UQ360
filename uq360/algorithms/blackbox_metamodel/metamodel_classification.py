import inspect
from collections import namedtuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from uq360.algorithms.posthocuq import PostHocUQ


class MetamodelClassification(PostHocUQ):
    """ Extracts confidence scores from black-box classification models using a meta-model [4]_ .

    References:
        .. [4] Chen, Tongfei, et al. "Confidence scoring using whitebox meta-models with linear classifier probes."
         The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.
    """

    def _create_named_model(self, mdltype, config):
        """ Instantiates a model by name passed in 'mdltype'.

        Args:
            mdltype: string with name (must be supported)
            config: dict with args passed in the instantiation call
        Returns:
             mdl instance
        """
        assert (isinstance(mdltype, str))
        if mdltype == 'lr':
            mdl = LogisticRegression(**config)
        elif mdltype == 'gbm':
            mdl = GradientBoostingClassifier(**config)
        else:
            raise NotImplementedError("ERROR: Requested model type unknown: \"%s\"" % mdltype)
        return mdl

    def _get_model_instance(self, model, config):
        """ Returns an instance of a model based on (a) a desired name or (b) passed in class, or
        (c) passed in instance.

        :param model: string, class, or instance. Class and instance must have certain methods callable.
        :param config: dict with args passed in during the instantiation
        :return: model instance
        """
        assert (model is not None and config is not None)
        if isinstance(model, str):  # 'model' is a name, create it
            mdl = self._create_named_model(model, config)
        elif inspect.isclass(model):  # 'model' is a class, instantiate it
            mdl = model(**config)
        else:  # 'model' is an instance, register it
            mdl = model
        if not all([hasattr(mdl, key) and callable(getattr(mdl, key)) for key in self.callable_keys]):
            raise ValueError("ERROR: Passed model/method failed the interface test. Methods required: %s" %
                             ','.join(self.callable_keys))
        return mdl

    def __init__(self, base_model=None, meta_model=None, base_config=None, meta_config=None, random_seed=42):
        """

        :param base_model: Base model. Can be:
                            (1) None (default mdl will be set up),
                            (2) Named model (e.g., logistic regression 'lr' or gradient boosting machine 'gbm'),
                            (3) Base model class declaration (e.g., sklearn.linear_model.LogisticRegression). Will instantiate.
                            (4) Model instance (instantiated outside). Will be re-used. Must have certain callable methods.
                            Note: user-supplied classes and models must have certain callable methods ('predict', 'fit')
                            and be capable of raising NotFittedError.
        :param meta_model: Meta model. Same values possible as with 'base_model'
        :param base_config: None or a params dict to be passed to 'base_model' at instantiation
        :param meta_config: None or a params dict to be passed to 'meta_model' at instantiation
        :param random_seed: seed used in the various pipeline steps
        """
        super(MetamodelClassification).__init__()
        self.random_seed = random_seed
        self.callable_keys = ['predict', 'fit']  # required methods - must be present in models passed in
        self.base_model_default = 'gbm'
        self.meta_model_default = 'lr'
        self.base_config_default = {'n_estimators': 300, 'max_depth': 10,
                                    'learning_rate': 0.001, 'min_samples_leaf': 10, 'min_samples_split': 10,
                                    'random_state': self.random_seed}
        self.meta_config_default = {'penalty': 'l1', 'C': 1, 'solver': 'liblinear', 'random_state': self.random_seed}
        self.base_config = base_config if base_config is not None else self.base_config_default
        self.meta_config = meta_config if meta_config is not None else self.meta_config_default
        self.base_model = None
        self.meta_model = None
        self.base_model = self._get_model_instance(base_model if base_model is not None else self.base_model_default,
                                                   self.base_config)
        self.meta_model = self._get_model_instance(meta_model if meta_model is not None else self.meta_model_default,
                                                   self.meta_config)

    def get_params(self, deep=True):
        return {"base_model": self.base_model, "meta_model": self.meta_model, "base_config": self.base_config,
                "meta_config": self.meta_config, "random_seed": self.random_seed}

    def _process_pretrained_model(self, X, y_hat_proba):
        """
        Given the original input features and the base output probabilities, generate input features
        to train a meta model. Current implementation copies all input features and appends.

        :param X: numpy [nsamples, dim]
        :param y_hat_proba: [nsamples, nclasses]
        :return: array with new features [nsamples, newdim]
        """
        assert (len(y_hat_proba.shape) == 2)
        assert (X.shape[0] == y_hat_proba.shape[0])
        # sort the probs sample by sample
        faux1 = np.sort(y_hat_proba, axis=-1)
        # add delta between top and second candidate
        faux2 = np.expand_dims(faux1[:, -1] - faux1[:, -2], axis=-1)
        return np.hstack([X, faux1, faux2])

    def fit(self, X, y, meta_fraction=0.2, randomize_samples=True, base_is_prefitted=False,
            meta_train_data=(None, None)):
        """
        Fit base and meta models.

        :param X: input to the base model,
                array-like of shape (n_samples, n_features).
                Features vectors of the training data.
        :param y: ground truth for the base model,
                array-like of shape (n_samples,)
        :param meta_fraction: float in [0,1] - a fractional size of the partition carved out to train the meta model
                                (complement will be used to train the base model)
        :param randomize_samples: use shuffling when creating partitions
        :param base_is_prefitted: Setting True will skip fitting the base model (useful for base models that have been
                        instantiated outside/by the user and are already fitted.
        :param meta_train_data: User supplied data to train the meta model. Note that this option should only be used
                        with 'base_is_prefitted'==True. Pass a tuple meta_train_data=(X_meta, y_meta) to activate.
                        Note that (X,y,meta_fraction, randomize_samples) will be ignored in this mode.
        :return: self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert (len(meta_train_data) == 2)
        if meta_train_data[0] is None:
            X_base, X_meta, y_base, y_meta = train_test_split(X, y, shuffle=randomize_samples, test_size=meta_fraction,
                                                              random_state=self.random_seed)
        else:
            if not base_is_prefitted:
                raise ValueError("ERROR: fit(): base model must be pre-fitted to use the 'meta_train_data' option")
            X_base = y_base = None
            X_meta = meta_train_data[0]
            y_meta = meta_train_data[1]
        # fit the base model
        if not base_is_prefitted:
            self.base_model.fit(X_base, y_base)
        # get input for the meta model from the base
        try:
            y_hat_meta_proba = self.base_model.predict_proba(X_meta)
            # determine correct-incorrect outcome - these are targets for the meta model trainer
            y_hat_meta_targets = np.asarray((y_meta == np.argmax(y_hat_meta_proba, axis=-1)), dtype=np.int)
        except NotFittedError as e:
            raise RuntimeError("ERROR: fit(): The base model appears not pre-fitted (%s)" % repr(e))
        # get input features for meta training
        X_meta_in = self._process_pretrained_model(X_meta, y_hat_meta_proba)
        # train meta model to predict 'correct' vs. 'incorrect' of the base
        self.meta_model.fit(X_meta_in, y_hat_meta_targets)
        return self

    def predict(self, X):
        """
        Generate a base prediction along with uncertainty/confidence for data X.

        :param X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
        :return: namedtuple: A namedtuple that holds

            y_pred: ndarray of shape (n_samples,)
                Predicted labels of the test points.
            y_score: ndarray of shape (n_samples,)
                Confidence score the test points.

        """
        y_hat_proba = self.base_model.predict_proba(X)
        y_hat = np.argmax(y_hat_proba, axis=-1)
        X_meta_in = self._process_pretrained_model(X, y_hat_proba)
        z_hat = self.meta_model.predict_proba(X_meta_in)
        index_of_class_1 = np.where(self.meta_model.classes_ == 1)[0][0]  # class 1 corresponds to probab of positive/correct outcome
        Result = namedtuple('res', ['y_pred', 'y_score'])
        res = Result(y_hat, z_hat[:, index_of_class_1])

        return res
