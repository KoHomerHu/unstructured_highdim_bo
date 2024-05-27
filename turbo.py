import torch
from torch.quasirandom import SobolEngine
import os

from botorch.models import SingleTaskGP
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval

from benchmarking.mappings import ACQUISITION_FUNCTIONS
from benchmarking.gp_priors import get_covar_module

from state_evol import TurboState
from ls_evol import TrustRegionEvol

"""
High-dimenensional Bayesian Optimization that combines vanilla BO and TuRBO.
1. Initialize the model with Sobol points.
2. Perform vanilla BO (with minor adjustment) until GSS criterion is met or the maximum number of iterations is reached.
3. Perform TuRBO-1 until convergence.

Parameters:
- model: the GP model to use during the vanilla BO stage
- model_kwargs: keyword arguments for the GP model
- acq_func: the acquisition function to use during the vanilla BO stage
- opt_kwargs: keyword arguments for the optimize_acqf method
- objective: the objective function to optimize
- bounds: the bounds of the input space
- num_init: the number of initial points to use for the Sobol initialization
- num_bo: the number of BO iterations to perform after the Sobol initialization
- num_turbo: the number of TuRBO iterations to perform after the vanilla BO stage
- device: the device to use for the optimization
- dtype: the data type to use for the optimization (default: torch.double)
- seed: the seed to use for the optimization (default: 0)
- gss: the global search strategy to use (default: None)
"""
class VanillaTuRBO(TrustRegionEvol):
    def __init__(self, model, model_params, acq_func, opt_kwargs, objective, bounds, num_init, num_bo, num_turbo, device, dtype=torch.double, seed=None, gss=None):
        self.model = model
        self.model_params = model_params
        self.acq_func = acq_func
        self.opt_kwargs = opt_kwargs
        self.objective = objective
        self.bounds = bounds
        self.dim = len(bounds.T) # dimension of the input space
        self.num_init = num_init
        self.num_bo = num_bo
        self.num_turbo = num_turbo
        self.device = device
        self.dtype  = dtype
        self.seed = seed
        self.state = TurboState(dim=len(bounds.T)) # Use TurboState to control the evolution of the TR

        assert gss in (None, 'lqr', 'ei', 'pi') # available global search strategies
        self.gss = gss

        self.X, self.y = None, None # unnormalized data points and evaluations

        self.results = {} # results of the optimization
        for i in range(self.dim):
            self.results[f"x_{i+1}"] = []
        self.results['Eval'] = []
        self.results['Best Value'] = []
        self.results['EI'] = [] # Records EI at the incumbent
        self.results['PI'] = [] # Records PI at the incumbent

        self.hyperparameters = {}

    """Global stopping criterion used during stage 2"""
    def should_stop(self):
        if self.gss is None:
            return False
        else:
            raise NotImplementedError

    """
    Given the current state and a GP model with data X and Y
    Use Thompson sampling to generate the next point in the trust region.
    Modified from the BoTorch tutorial (https://botorch.org/tutorials/turbo_1).
    """
    def generate_next_point(self, state, model, X, y):
        x_center, tr_lb, tr_ub = self.state.get_trust_region(model, X, y)

        # First generate a scrambled Sobol sequence within the TR
        dim = X.shape[-1]
        n_candidates = min(5000, 200 * dim)
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=X.dtype, device=X.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Use the value in the Sobol sequence with a certain probability
        pert_prob = min(1.0, 20 / dim) # not perterb all coordinates at once for high-dim functions
        mask = torch.rand(n_candidates, dim, dtype=X.dtype, device=X.device) <= pert_prob
        ind = torch.where(mask.sum(dim=1)==0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

        # Create the candidate set as masked perturbations
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            next_X = thompson_sampling(X_cand, num_samples=1)
        
        return next_X

    """Perform vanilla BO with the specified model and acquisition function."""
    def vanilla_bo_stage(self, save_file=None):
        model_kwargs = get_covar_module(**self.model_params) 

        for iter in range(self.num_bo):
            # Normalize parameters and standariize evaluations
            train_X = normalize(self.X, bounds=self.bounds)
            train_y = (self.y - self.y.mean()) / self.y.std()

            # Initialize the model and likelihood
            covar_module = model_kwargs['covar_module_class'](**model_kwargs['covar_module_options'])
            likelihood = model_kwargs['likelihood_class'](**model_kwargs['likelihood_options'])
            model = self.model(train_X, train_y, covar_module=covar_module, likelihood=likelihood)
            # model = self.model(train_X, train_y, likelihood=likelihood) # test without scaling the parameters for lengthscales' prior
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Optimize the acquisition function
            if self.acq_func == ACQUISITION_FUNCTIONS['qLogNEI']:
                acf = self.acq_func(model, train_X, prune_baseline=True) # Check ./configs/acq/qlognei.yaml for specifications
            else:
                raise ValueError(f"Acquisition function {self.acq_func} is not available - feel free to add it!")
            candidate, _ = optimize_acqf(
                acf, 
                bounds=torch.stack(
                    [
                        torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                        torch.ones(self.dim, dtype=self.dtype, device=self.device)
                    ]
                ),
                q=1, # generate only 1 candidate instead of a batch
                **self.opt_kwargs
            )
            
            # Update X and y
            next_X = unnormalize(candidate[0], bounds=self.bounds)
            next_y = torch.tensor(
                [self.eval_objective(candidate[0]),], 
                dtype=self.dtype, 
                device=self.device
            ).unsqueeze(-1)
            self.X = torch.cat([self.X, next_X.unsqueeze(0)], dim=0)
            self.y = torch.cat([self.y, next_y], dim=0)

            # Update the results and calculate EI and PI at the incumbent
            for j in range(self.dim):
                self.results[f"x_{j+1}"].append(next_X[j].cpu().item())
            self.results['Eval'].append(next_y.cpu().item())
            self.results['Best Value'].append(self.y.max().cpu().item())
            ei_func = ExpectedImprovement(model, train_y.max())
            ei = ei_func(candidate[0].unsqueeze(0)).cpu().item()
            self.results['EI'].append(ei)
            pi_func = ProbabilityOfImprovement(model, train_y.max())
            pi = pi_func(candidate[0].unsqueeze(0)).cpu().item()
            self.results['PI'].append(pi)

            print(f"(2) Iteration {self.num_init+iter+1}/{self.num_init+self.num_bo}: ({', '.join([f'{x:.2f}' for x in next_X.tolist()])}) -> {next_y.cpu().item():.2f} with (EI: {ei:.2f}, PI: {pi:.2f}, Best Value: {self.y.max().item():.2f})")

            self.save_results(save_file, model=model) # Save the results

            # Check if the global stopping criterion is met
            if self.should_stop():
                break       

    """Perform TuRBO-1 with Thompson Sampling after the vanilla BO stage."""
    def turbo_stage(self, save_file=None):
        state = TurboState(dim=self.dim)

        turbo_iter = 0
        while not state.restart_triggered:
            if self.num_turbo > 0 and turbo_iter >= self.num_turbo:
                break
            # Normalize parameters and standariize evaluations
            train_X = normalize(self.X, bounds=self.bounds)
            train_y = (self.y - self.y.mean()) / self.y.std()

            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )
            model = SingleTaskGP(train_X, train_y, likelihood=likelihood, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Always use Cholesky
            with gpytorch.settings.max_cholesky_size(float("inf")): 
                # Fit the GP model
                fit_gpytorch_mll(mll)

                # Generate the next point in the TR using Thompson Sampling
                raw_next_X = self.generate_next_point(state, model, train_X, train_y)
                next_X = unnormalize(raw_next_X, bounds=self.bounds).squeeze(0)
                next_y = torch.tensor(
                    [self.eval_objective(raw_next_X)],
                    dtype=self.dtype,
                    device=self.device
                ).unsqueeze(-1)

                # Append the data points and evaluations
                self.X = torch.cat([self.X, next_X.unsqueeze(0)], dim=0)
                self.y = torch.cat([self.y, next_y], dim=0)

                # Update state
                state.update_state(next_y)

                # Update the results
                for j in range(self.dim):
                    self.results[f"x_{j+1}"].append(next_X[j].cpu().item())
                self.results['Eval'].append(next_y.cpu().item())
                self.results['Best Value'].append(self.y.max().cpu().item())
                self.results['EI'].append(float('nan'))
                self.results['PI'].append(float('nan'))

                print(f"(3) Iteration {self.num_init+self.num_bo+turbo_iter+1}/{self.num_init+self.num_bo+self.num_turbo}: ({', '.join([f'{x:.2f}' for x in next_X.tolist()])}) -> {next_y.cpu().item():.2f} with (L: {state.length:.2e}, Best Value: {self.y.max().item():.2f})")

                self.save_results(save_file) # Save the results

            turbo_iter += 1

    def run(self, save_path=None, save_name=None):
        # Get the address to save the results
        if save_path is not None and save_name is not None:
            os.makedirs(save_path, exist_ok=True)
            save_file = f"{save_path}/{save_name}.csv"
        else:
            save_file = None

        self.sobol_stage(save_file)
        self.vanilla_bo_stage(save_file)
        self.turbo_stage(save_file)