import autograd
import autograd.numpy as np
import scipy.optimize
from autograd import grad
from autograd.scipy.special import logsumexp
from sklearn.cluster import KMeans


class HMM:
    """
    A Hidden Markov Model with Gaussian observations with
    unknown means and known precisions.
    """
    def __init__(self, X, config_dict=None):
        self.N, self.T, self.D = X.shape
        self.K = config_dict['K']  # number of HMM states
        self.I = np.eye(self.K)
        self.Precision = np.zeros([self.D, self.D, self.K])
        self.X = X
        if config_dict['precision'] is None:
            for k in np.arange(self.K):
                self.Precision[:, :, k] = np.eye(self.D)
        else:
                self.Precision = config_dict['precision']
        self.dParams_dWeights = None
        self.alphaT = None # Store the final beliefs.
        self.beta1 = None # store the first timestep beliefs from the beta recursion.
        self.forward_trellis = {}  # stores \alpha
        self.backward_trellis = {}  # stores \beta

    def initialize_params(self, seed=1234):
        np.random.seed(seed)
        param_dict = {}
        A = np.random.randn(self.K, self.K)
        # use k-means to initialize the mean parameters
        X = self.X.reshape([-1, self.D])
        kmeans = KMeans(n_clusters=self.K, random_state=seed,
                        n_init=15).fit(X)
        labels = kmeans.labels_
        _, counts = np.unique(labels, return_counts=True)
        pi = counts
        phi = kmeans.cluster_centers_

        param_dict['A'] = np.exp(A)
        param_dict['pi0'] = pi
        param_dict['phi'] = phi
        return self.pack_params(param_dict)

    def unpack_params(self, params):
        param_dict = dict()
        K = self.K
        # For unpacking simplex parameters: have packed them as
        # log(pi[:-1]) - log(pi[-1]).
        unnorm_A = np.exp(np.append(params[:K**2-K].reshape(K, K-1),
                                    np.zeros((K, 1)),
                                    axis=1)
                          )
        Z = np.sum(unnorm_A[:, :-1], axis=1)
        unnorm_A /= Z[:, np.newaxis]
        norm_A = unnorm_A / unnorm_A.sum(axis=1, keepdims=True)
        param_dict['A'] = norm_A

        unnorm_pi = np.exp(np.append(params[K**2-K:K**2-1], 0.0))
        Z = np.sum(unnorm_pi[:-1])
        unnorm_pi /= Z
        param_dict['pi0'] = unnorm_pi / unnorm_pi.sum()
        param_dict['phi'] = params[K**2-K+K-1:].reshape(self.D, K)
        return param_dict

    def weighted_alpha_recursion(self, xseq, pi, phi, Sigma, A, wseq, store_belief=False):
        """
        Computes the weighted marginal probability of the sequence xseq given parameters;
        weights wseq turn on or off the emissions p(x_t | z_t) (weighting scheme B)
        :param xseq: T * D
        :param pi: K * 1
        :param phi: D * K
        :param wseq: T * 1
        :param A:
        :return:
        """
        ll = self.log_obs_lik(xseq[:, :, np.newaxis], phi[np.newaxis, :, :], Sigma)
        alpha = np.log(pi.ravel()) + wseq[0] * ll[0]
        if wseq[0] == 0:
            self.forward_trellis[0] = alpha[:, np.newaxis]
        for t in np.arange(1, self.T):
            alpha = logsumexp(alpha[:, np.newaxis] + np.log(A), axis=0) + wseq[t] * ll[t]
            if wseq[t] == 0:
                # store the trellis, would be used to compute the posterior z_t | x_1...x_t-1, x_t+1, ...x_T
                self.forward_trellis[t] = alpha[:, np.newaxis]
        if store_belief:
            # store the final belief
            self.alphaT = alpha
        return logsumexp(alpha)

    def weighted_beta_recursion(self, xseq, pi, phi, Sigma, A, wseq, store_belief=False):
            """
            Runs beta recursion;
            weights wseq turn on or off the emissions p(x_t | z_t) (weighting scheme B)
            :param xseq: T * D
            :param pi: K * 1
            :param phi: D * K
            :param wseq: T * 1
            :param A:
            :return:
            """
            ll = self.log_obs_lik(xseq[:, :, np.newaxis], phi[np.newaxis, :, :], Sigma)
            beta = np.zeros_like(pi.ravel())  # log(\beta) of all ones.
            max_t = ll.shape[0]
            if wseq[max_t - 1] == 0:
                # store the trellis, would be used to compute the posterior z_t | x_1...x_t-1, x_t+1, ...x_T
                self.backward_trellis[max_t - 1] = beta[:, np.newaxis]
            for i in np.arange(1, max_t):
                t = max_t - i - 1
                beta = logsumexp((beta + wseq[t + 1] * ll[t + 1])[np.newaxis, :] + np.log(A), axis=1)
                if wseq[t] == 0:
                    # store the trellis, would be used to compute the posterior z_t | x_1...x_t-1, x_t+1, ...x_T
                    self.backward_trellis[t] = beta[:, np.newaxis]
            # account for the init prob
            beta = (beta + wseq[0] * ll[0]) + np.log(pi.ravel())
            if store_belief:
                # store the final belief
                self.beta1 = beta
            return logsumexp(beta)

    def weighted_loss(self, params, weights):
        """
        For LOOCV / IF computation within a single sequence. Uses weighted alpha recursion
        :param params:
        :param weights:
        :return:
        """
        param_dict = self.unpack_params(params)
        logp = self.get_prior_contrib(param_dict)
        logp = logp + self.weighted_alpha_recursion(self.X[0], param_dict['pi0'],
                                                               param_dict['phi'],
                                                               self.Precision,
                                                               param_dict['A'],
                                                               weights)
        return -logp

    def loss_at_missing_timesteps(self, weights, params):
        """
        :param weights: zeroed out weights indicate missing values
        :param params: packed parameters
        :return:
        """
        # empty forward and backward trellis
        self.clear_trellis()
        param_dict = self.unpack_params(params)
        # populate forward and backward trellis
        lpx = self.weighted_alpha_recursion(self.X[0], param_dict['pi0'],
                                                  param_dict['phi'],
                                                  self.Precision,
                                                  param_dict['A'],
                                                  weights,
                                                  store_belief=True )
        lpx_alt = self.weighted_beta_recursion(self.X[0], param_dict['pi0'],
                                                 param_dict['phi'],
                                                 self.Precision,
                                                 param_dict['A'],
                                                 weights,
                                                 store_belief=True)
        assert np.allclose(lpx, lpx_alt) # sanity check
        test_ll = []
        # compute loo likelihood
        ll = self.log_obs_lik(self.X[0][:, :, np.newaxis], param_dict['phi'], self.Precision)
        # compute posterior p(z_t | x_1,...t-1, t+1,...T) \forall missing t
        tsteps = []
        for t in self.forward_trellis.keys():
            lpz_given_x = self.forward_trellis[t] + self.backward_trellis[t] - lpx
            test_ll.append(logsumexp(ll[t] + lpz_given_x.ravel()))
            tsteps.append(t)
        # empty forward and backward trellis
        self.clear_trellis()
        return -np.array(test_ll)

    def fit(self, weights, init_params=None, num_random_restarts=1, verbose=False, maxiter=None):
        if maxiter:
            options_dict = {'disp': verbose, 'gtol': 1e-10, 'maxiter': maxiter}
        else:
            options_dict = {'disp': verbose, 'gtol': 1e-10}

        # Define a function that returns gradients of training loss using Autograd.
        training_loss_fun = lambda params: self.weighted_loss(params, weights)
        training_gradient_fun = grad(training_loss_fun, 0)
        if init_params is None:
            init_params = self.initialize_params()
        if verbose:
            print("Initial loss: ", training_loss_fun(init_params))
        res = scipy.optimize.minimize(fun=training_loss_fun,
                                      jac=training_gradient_fun,
                                      x0=init_params,
                                      tol=1e-10,
                                      options=options_dict)
        if verbose:
            print('grad norm =', np.linalg.norm(res.jac))
        return res.x

    def clear_trellis(self):
        self.forward_trellis = {}
        self.backward_trellis = {}

    #### Required for IJ computation ###
    def compute_hessian(self, params_one, weights_one):
        return autograd.hessian(self.weighted_loss, argnum=0)(params_one, weights_one)

    def compute_jacobian(self, params_one, weights_one):
        return autograd.jacobian(autograd.jacobian(self.weighted_loss, argnum=0), argnum=1)\
                                (params_one, weights_one).squeeze()
    ###################################################

    @staticmethod
    def log_obs_lik(x, phi, Sigma):
        """
        :param x: T*D*1
        :param phi: 1*D*K
        :param Sigma: D*D*K --- precision matrices per state
        :return: ll
        """
        centered_x = x - phi
        ll = -0.5 * np.einsum('tdk, tdk, ddk -> tk', centered_x, centered_x, Sigma )
        return ll

    @staticmethod
    def pack_params(params_dict):
        param_list = [(np.log(params_dict['A'][:, :-1]) -
                       np.log(params_dict['A'][:, -1])[:, np.newaxis]).ravel(),
                       np.log(params_dict['pi0'][:-1]) - np.log(params_dict['pi0'][-1]),
                       params_dict['phi'].ravel()]
        return np.concatenate(param_list)

    @staticmethod
    def get_prior_contrib(param_dict):
        logp = 0.0
        # Prior
        logp += -0.5 * (np.linalg.norm(param_dict['phi'], axis=0) ** 2).sum()
        logp += (1.1 - 1) * np.log(param_dict['A']).sum()
        logp += (1.1 - 1) * np.log(param_dict['pi0']).sum()
        return logp

    @staticmethod
    def get_indices_in_held_out_fold(T, pct_to_drop, contiguous=False):
        """
        :param T: length of the sequence
        :param pct_to_drop: % of T in the held out fold
        :param contiguous: if True generate a block of indices to drop else generate indices by iid sampling
        :return: o (the set of indices in the fold)
        """
        if contiguous:
            l = np.floor(pct_to_drop / 100. * T)
            anchor = np.random.choice(np.arange(l + 1, T))
            o = np.arange(anchor - l, anchor).astype(int)
        else:
            # i.i.d LWCV
            o = np.random.choice(T - 2, size=np.int(pct_to_drop / 100. * T), replace=False) + 1
        return o

    @staticmethod
    def synthetic_hmm_data(K, T, D, sigma0=None, seed=1234, varainces_of_mean=1.0,
                        diagonal_upweight=False):
        """
        :param K: Number of HMM states
        :param T: length of the sequence
        """
        N = 1 # For structured IJ we will remove data / time steps from a single sequence
        np.random.seed(seed)
        if sigma0 is None:
            sigma0 = np.eye(D)

        A = np.random.dirichlet(alpha=np.ones(K), size=K)
        if diagonal_upweight:
            A = A + 3 * np.eye(K)  # add 3 to the diagonal and renormalize to encourage self transitions
            A = A / A.sum(axis=1)

        pi0 = np.random.dirichlet(alpha=np.ones(K))
        mus = np.random.normal(size=(K, D), scale=np.sqrt(varainces_of_mean))
        zs = np.empty((N, T), dtype=np.int)
        X = np.empty((N, T, D))

        for n in range(N):
            zs[n, 0] = int(np.random.choice(np.arange(K), p=pi0))
            X[n, 0] = np.random.multivariate_normal(mean=mus[zs[n, 0]], cov=sigma0)
            for t in range(1, T):
                zs[n, t] = int(np.random.choice(np.arange(K), p=A[zs[n, t - 1], :]))
                X[n, t] = np.random.multivariate_normal(mean=mus[zs[n, t]], cov=sigma0)

        return {'X': X, 'state_assignments': zs, 'A': A, 'initial_state_assignment': pi0,  'means': mus}
