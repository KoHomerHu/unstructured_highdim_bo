import torch
from torch.quasirandom import SobolEngine
from scipy.stats import expectile

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


class SigmoidBO:
    def __init__(self, model, model_params, acq_func, opt_kwargs, objective, bounds, num_init, num_bo, device, dtype=torch.double, seed=None):
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

        self.X, self.y = None, None # unnormalized data points and evaluations

        self.results = {} # results of the optimization
        for i in range(self.dim):
            self.results[f"x_{i+1}"] = []
        self.results['Eval'] = []
        self.results['Best Value'] = []
        self.results['EI'] = [] # Records EI at the incumbent
        self.results['PI'] = [] # Records PI at the incumbent

        self.hyperparameters = {}

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
            [self.eval_objective(x, seed=self.seed) for x in raw_X], 
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

        print(f"(1) Sobol Initialization Completed. {self.num_init} points generated. Best value: {self.results['Best Value'][-1]:.2f}")

        self.save_results(save_file) # Save the results

    def vanilla_bo_stage(self, save_file=None, k=1, c=1.5, bootstrapping=False):
        def sigma_k(y):
            return c * (y/c) / (1 + abs((y/c))**k) ** (1/k) 
        
        """
        Use bootstrapping to estimate more robust mean and standard deviation that is insensitive to concentration of inputs.
        """
        def estimate_mean_std(X, y):
            # Fit another surrogate model
            model_kwargs = get_covar_module(**self.model_params)
            covar_module = model_kwargs['covar_module_class'](**model_kwargs['covar_module_options'])
            likelihood = model_kwargs['likelihood_class'](**model_kwargs['likelihood_options'])
            model = self.model(X, y, covar_module=covar_module, likelihood=likelihood).to(self.device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Generate N = 1024 sobol points to sample from the posterior
            with torch.no_grad(): # We don't need gradients for sampling
                N = 1024
                sobol = SobolEngine(
                    dimension=self.dim, 
                    scramble=True, 
                    seed=self.seed
                )
                sample_inputs = sobol.draw(n=N).to(dtype=self.dtype, device=self.device)
                posterior = model.posterior(sample_inputs)
                samples = posterior.rsample(sample_shape=torch.Size([N]))
                mean = samples.mean()
                std = samples.std() * (N / (N - 1)) ** 0.5 # Correct the std for sampling error
            return mean, std

        bo_iter = 0
        while bo_iter < self.num_bo:
            # Normalize parameters and simplify evaluations
            train_X = normalize(self.X, bounds=self.bounds)
            train_y = (self.y - self.y.mean()) / self.y.std() if self.y.std() > 0  else (self.y - self.y.mean()) # standardize
            original_y_max = train_y.max().item()
            original_y_min = train_y.min().item()
            if bootstrapping:
                est_mean, est_std = estimate_mean_std(train_X, train_y)
                train_y = (train_y - est_mean) / est_std # re-standardize based on bootstrapping
            train_y = train_y.apply_(sigma_k) # simplification via soft winsorization
            # if bootstrapping:
            #     train_y = (train_y - expectile(train_y, alpha=0.95)) / train_y.std() # re-standardize asymmetrically
            # else:
            train_y = (train_y - train_y.mean()) / train_y.std() if train_y.std() > 0  else (train_y - train_y.mean())
            new_y_max = train_y.max().item()
            new_y_min = train_y.min().item()

            # Train the GP model globally to get the TR
            model_kwargs = get_covar_module(**self.model_params) 
            covar_module = model_kwargs['covar_module_class'](**model_kwargs['covar_module_options'])
            likelihood = model_kwargs['likelihood_class'](**model_kwargs['likelihood_options'])
            model = self.model(train_X, train_y, covar_module=covar_module, likelihood=likelihood).to(self.device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Generate the next point in the TR using EI
            raw_next_X = self.generate_next_point(model, train_X, bounds=None) # No trust region
            next_X = unnormalize(raw_next_X, bounds=self.bounds).squeeze(0)
            next_y = torch.tensor(
                [self.eval_objective(raw_next_X, seed=self.seed+self.num_init+bo_iter)],
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
            ei = ei_func(raw_next_X).cpu().item()
            self.results['EI'].append(ei)
            pi_func = ProbabilityOfImprovement(model, train_y.max())
            pi = pi_func(raw_next_X).cpu().item()
            self.results['PI'].append(pi)

            print(f"(2) Iteration {self.num_init+bo_iter+1}/{self.num_init+self.num_bo}: ({', '.join([f'{x:.2f}' for x in next_X.tolist()])}) -> {next_y.cpu().item():.2f} with (old_y_range: ({original_y_min:.2f}, {original_y_max:.2f}), new_y_range: ({new_y_min:.2f}, {new_y_max:.2f}), EI: {ei:.2f}, PI: {pi:.2f}, Best Value: {self.results['Best Value'][-1]:.2f})")

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
    


