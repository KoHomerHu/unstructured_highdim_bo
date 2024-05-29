import torch
from torch.quasirandom import SobolEngine

import json
import os
import pandas as pd

from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood

# from benchmarking.eval_utils import get_model_hyperparameters
from benchmarking.mappings import ACQUISITION_FUNCTIONS
from benchmarking.gp_priors import get_covar_module
from benchmarking.eval_utils import get_model_hyperparameters

from state_evol import (
    DummyState, 
    TurboState,
    AlphaRatioState,
    EIThresholdState,
    PIThresholdState,
)

"""
High-dimenensional Bayesian Optimization that uses vanilla BO with evolving trust region.
1. Initialize the model with Sobol points.
2. Perform vanilla BO in the trust region controled by the evol_state_maintainer

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
- evol_state_maintainer: the evolution strategy to use (default: DummyState, i.e. equivalent to vanilla BO)
"""
class TrustRegionEvol:
    def __init__(self, model, model_params, acq_func, opt_kwargs, objective, bounds, num_init, num_bo, device, dtype=torch.double, seed=None, evol_state_maintainer=DummyState):
        self.model = model
        self.model_params = model_params
        self.acq_func = acq_func
        self.opt_kwargs = opt_kwargs
        self.objective = objective
        self.bounds = bounds
        self.dim = len(bounds.T) # dimension of the input space
        self.num_init = num_init
        self.num_bo = num_bo
        self.device = device
        self.dtype  = dtype
        self.seed = seed
        self.state = evol_state_maintainer(dim=len(bounds.T)) 

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

    def generate_next_point(self, model, X, bounds=None):
        # Optimize the acquisition function
        if self.acq_func == ACQUISITION_FUNCTIONS['qLogNEI']:
            acf = self.acq_func(model, X, prune_baseline=True) # Check ./configs/acq/qlognei.yaml for specifications
        else:
            raise ValueError(f"Acquisition function {self.acq_func} is not available - feel free to add it!")
        
        if bounds is None:
            bounds = torch.stack(
                [
                    torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                    torch.ones(self.dim, dtype=self.dtype, device=self.device)
                ]
            )

        next_X, _ = optimize_acqf(
            acf, 
            bounds=bounds,
            q=1, # generate only 1 candidate instead of a batch
            **self.opt_kwargs
        )

        return next_X
    
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

        self.state.best_value = self.y.max().cpu().item()

        self.save_results(save_file) # Save the results

    def vanilla_bo_stage(self, save_file=None):
        bo_iter = 0

        while not self.state.restart_triggered and bo_iter < self.num_bo:
            # Normalize parameters and standariize evaluations
            train_X = normalize(self.X, bounds=self.bounds)
            train_y = (self.y - self.y.mean()) / self.y.std()

            # Train the GP model globally to get the TR
            model_kwargs = get_covar_module(**self.model_params, side_length=self.state.length) 
            covar_module = model_kwargs['covar_module_class'](**model_kwargs['covar_module_options'])
            likelihood = model_kwargs['likelihood_class'](**model_kwargs['likelihood_options'])
            model = self.model(train_X, train_y, covar_module=covar_module, likelihood=likelihood).to(self.device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            _, tr_lb, tr_ub = self.state.get_trust_region(model, train_X, train_y)

            # Generate the next point in the TR using EI
            tr_bounds = torch.stack([tr_lb, tr_ub]).to(self.device)
            raw_next_X = self.generate_next_point(model, train_X, bounds=tr_bounds) 
            next_X = unnormalize(raw_next_X, bounds=self.bounds).squeeze(0)
            next_y = torch.tensor(
                [self.eval_objective(raw_next_X)],
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(-1)

            # Append the data points and evaluations
            self.X = torch.cat([self.X, next_X.unsqueeze(0)], dim=0)
            self.y = torch.cat([self.y, next_y], dim=0)

            # Update the results
            for j in range(self.dim):
                self.results[f"x_{j+1}"].append(next_X[j].cpu().item())
            self.results['Eval'].append(next_y.cpu().item())
            self.results['Best Value'].append(self.y.max().cpu().item())

            ei_func = ExpectedImprovement(model, train_y.max())
            ei = ei_func(next_X.unsqueeze(0)).cpu().item()
            self.results['EI'].append(ei)
            pi_func = ProbabilityOfImprovement(model, train_y.max())
            pi = pi_func(next_X.unsqueeze(0)).cpu().item()
            self.results['PI'].append(pi)

            # Update state
            if isinstance(self.state, EIThresholdState):
                self.state.update_state(next_y, ei)
            elif isinstance(self.state, PIThresholdState):
                self.state.update_state(next_y, pi)
            elif isinstance(self.state, AlphaRatioState):
                self.state.update_state(next_y, model, train_X, train_y, self.opt_kwargs)
            else:
                self.state.update_state(next_y)

            print(f"(2) Iteration {self.num_init+bo_iter+1}/{self.num_init+self.num_bo}: ({', '.join([f'{x:.2f}' for x in next_X.tolist()])}) -> {next_y.cpu().item():.2f} with (L: {self.state.length:.2e}, EI: {ei:.2f}, PI: {pi:.2f}, Best Value: {self.y.max().item():.2f})")

            self.save_results(save_file, model) # Save the results

            bo_iter += 1

    def run(self, save_path=None, save_name=None):
        # Get the address to save the results
        if save_path is not None and save_name is not None:
            os.makedirs(save_path, exist_ok=True)
            save_file = f"{save_path}/{save_name}.csv"
        else:
            save_file = None

        self.sobol_stage(save_file)
        self.vanilla_bo_stage(save_file)

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
