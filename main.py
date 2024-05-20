from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from omegaconf import DictConfig
from os.path import dirname, join
import pandas as pd
import numpy as np
import math
import json
import os
import hydra
import torch
from torch.quasirandom import SobolEngine

from benchmarking.eval_utils import get_model_hyperparameters
from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS
)
from benchmarking.gp_priors import (
    MODELS,
    get_covar_module
)
from turbo import TurboState, generate_next_point

"""
High-dimenensional Bayesian Optimization without considering low-dimensional structures.
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
- device: the device to use for the optimization
- dtype: the data type to use for the optimization (default: torch.double)
- seed: the seed to use for the optimization (default: 0)
- gss: the global search strategy to use (default: None)
"""
class UnstructuredBO:
    def __init__(self, model, model_kwargs, acq_func, opt_kwargs, objective, bounds, num_init, num_bo, num_turbo, device, dtype=torch.double, seed=None, gss=None):
        self.model = model
        self.model_kwargs = model_kwargs
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

    """Unnormalize the input and then evaluate the objective function"""
    def eval_objective(self, x, seed=None):
        parameters = unnormalize(x, bounds=self.bounds)
        if seed is not None:
            return self.objective.evaluate_true(parameters, seed=seed)
        else:
            return self.objective(parameters)

    """Global stopping criterion used during stage 2"""
    def should_stop(self):
        if self.gss is None:
            return False
        else:
            raise NotImplementedError

    """Generate num_init of Sobol points."""
    def sobol_stage(self, save_file=None):
        sobol = SobolEngine(
            dimension=self.dim, 
            scramble=True, 
            seed=self.seed
        )

        # Generate the Sobol points and get evaluations
        raw_X = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
        self.X = unnormalize(raw_X, bounds=self.bounds)
        self.y = torch.tensor(
            [self.eval_objective(x) for x in raw_X], 
            dtype=self.dtype, 
            device=self.device
        ).unsqueeze(-1)

        # Update the results
        for i in range(self.num_init):
            for j in range(self.dim):
                self.results[f"x_{j+1}"].append(self.X[i][j].item())
            self.results['Eval'].append(self.y[i].item())
            self.results['Best Value'].append(self.y[:i+1].max().item())
            self.results['EI'].append(float('nan'))
            self.results['PI'].append(float('nan'))

        print(f"(1) Sobol Initialization Completed. {self.num_init} points generated. Best value: {self.y.max().item():.2f}")

        self.save_results(save_file) # Save the results

    """Perform vanilla BO with the specified model and acquisition function."""
    def vanilla_bo_stage(self, save_file=None):
        for iter in range(self.num_bo):
            # Normalize parameters and standariize evaluations
            train_X = normalize(self.X, bounds=self.bounds)
            train_y = (self.y - self.y.mean()) / self.y.std()

            # Initialize the model and likelihood
            covar_module = self.model_kwargs['covar_module_class'](**self.model_kwargs['covar_module_options'])
            likelihood = self.model_kwargs['likelihood_class'](**self.model_kwargs['likelihood_options'])
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
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        import gpytorch
        from botorch.models import SingleTaskGP

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
                raw_next_X = generate_next_point(state, model, train_X, train_y)
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
    
    def save_results(self, save_file, model=None):
        if save_file is not None:
            # Save X, y, best value, EI, and PI
            results_df = pd.DataFrame.from_dict(self.results)
            results_df.to_csv(save_file)
            # Save hyperparameters during the vanilla BO stage
            if model is not None and self.num_init < len(results_df) <= self.num_init + self.num_bo:
                self.hyperparameters[f'iter_{len(results_df)}'] = get_model_hyperparameters(model, self.y)
                with open(f"{save_file[:-4]}_hps.json", "w") as f:
                    json.dump(self.hyperparameters, f, indent=2)

        

@hydra.main(config_path='./configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the bounds of the input space
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0).to(device)

    # Get the model and model_kwargs
    model = MODELS[cfg.model.gp]
    model_kwargs = get_covar_module(
        cfg.model.model_name, 
        len(bounds.T), 
        gp_params=cfg.model.get('gp_params', None),
        gp_constraints=cfg.model.get('gp_constraints', {})
    )

    # If the acquisition function is Sampling, set num_init to the number of iterations
    # Otherwise, set num_init to the calculated number of initial points
    if cfg.acq.name == 'Sampling':
        num_init = cfg.benchmark.num_iters
    if cfg.init == "sqrt":
        factor = cfg.init_factor
        num_init = math.ceil(factor * len(cfg.benchmark.bounds) ** 0.5)
    elif isinstance(cfg.benchmark.num_init, int):
        num_init = max(cfg.benchmark.num_init, cfg.q)
    
    # Calculate the maximum number of vanilla BO iterations
    num_bo = cfg.benchmark.num_iters - num_init
    # Get the maximum number of TuRBO iterations
    num_turbo = cfg.benchmark.num_turbo if hasattr(cfg.benchmark, 'num_turbo') else -1 # default to -1, i.e. run until TuRBO converges
    # Get the test function as objective
    if hasattr(cfg.benchmark, 'outputscale'):
        test_function = get_test_function(
            name=cfg.benchmark.name, 
            noise=float(cfg.benchmark.noise_std), 
            seed=cfg.seed, 
            bounds=float(cfg.benchmark.outputscale)
        )
    else:
        test_function = get_test_function(
            name=cfg.benchmark.benchmark,
            noise_std=float(cfg.benchmark.noise_std),
            seed=cfg.seed,
            bounds=cfg.benchmark.bounds,
        )

    # initialize the optimizer
    optimizer = UnstructuredBO(
        model=model,
        model_kwargs=model_kwargs,
        acq_func=ACQUISITION_FUNCTIONS[cfg.acq.acq_func],
        opt_kwargs=dict(cfg.acq_opt),
        objective=test_function,
        bounds=bounds,
        num_init=num_init,
        num_bo=num_bo,
        num_turbo=num_turbo,
        device=device,
        seed=cfg.seed,
        gss=cfg.gss if hasattr(cfg, 'gss') else None
    )

    # run the optimization
    optimizer.run(
        save_path=join(dirname(os.path.abspath(__file__)), cfg.result_path),
        save_name=cfg.experiment_name
    )

if __name__ == '__main__':
    main()
