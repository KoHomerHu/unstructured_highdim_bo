import math
import torch
import copy

import gpytorch

from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll


"""
A dummy state maintainer that fix the trust region length to 1.0 (hence equivalent to vanilla BO)
"""
class DummyState:
    def __init__(self, dim):
        self.dim = dim
        self.length = 1.0
        self.length_min = 0.0
        self.best_value = -float('inf')
        self.restart_triggered = False

    def update_state(self, next_y):
        self.best_value = max(self.best_value, max(next_y).item())

        if self.length < self.length_min:
            self.restart_triggered = True

    def get_trust_region(self, model, X, y):
        # Consider the entire domain before the lengthscales start to shrink
        if self.length == 1.0:
            return 0.5 * torch.ones(self.dim), torch.zeros(self.dim), torch.ones(self.dim)
        
        # Take the best observation as the center of TR
        x_center = X[y.argmax(), :].clone()

        # Rescale the TR side length according to the lengthscales in the GP model
        try:
            lengthscales = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except:
            lengthscales = model.covar_module.lengthscale.squeeze().detach()
        # lengthscales = lengthscales / lengthscales.mean()
        scale_factor = lengthscales / torch.prod(lengthscales.pow(1.0 / len(lengthscales)))

        # Define the trust region
        tr_lb = torch.clamp(x_center - scale_factor * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + scale_factor * self.length / 2.0, 0.0, 1.0)

        return x_center, tr_lb, tr_ub
    

"""
Use a fixed scheduler to evolve the side length.
"""
class SchedulerState(DummyState):
    def __init__(self, dim):
        super().__init__(dim)
        self.length = 1.0
        self.length_min = 0.5**7 / self.dim
        self.counter = 0
        self.total_num = math.ceil(10 * self.dim ** 0.5)

    def update_state(self, next_y):
        super().update_state(next_y)

        self.counter += 1
        if self.counter >= self.total_num:
            self.length *= 0.7
            self.counter = 0

        print("Length: ", self.length, " Best: ", self.best_value)


"""
Use EI threshold to control the TR side length.
"""
class EIThresholdState(DummyState):
    def __init__(self, dim):
        super().__init__(dim)
        self.threshold = 1e-5
        self.length = 1.0
        self.length_min = 0.5**7 / self.dim # need to tune

    def update_state(self, next_y, ei):
        super().update_state(next_y)

        if ei <= self.threshold:
            self.length *= 0.8

        print("EI: ", ei, " Length: ", self.length, " Best: ", self.best_value)


"""
Use PI threshold to control the TR side length.
"""
class PIThresholdState(DummyState):
    def __init__(self, dim):
        super().__init__(dim)
        self.threshold = 1e-3
        self.length = 1.0
        self.length_min = 0.5**7 / self.dim # need to tune

    def update_state(self, next_y, pi):
        super().update_state(next_y)

        if pi <= self.threshold:
            self.length *= 0.8

        print("PI: ", pi, " Length: ", self.length, " Best: ", self.best_value)
    

"""
Use alpha ratio cooling to control the TR side length.
ac_func takes log EI by default
"""
class AlphaRatioState(DummyState):
    def __init__(self, dim, alpha=1.0, acq_func=qLogNoisyExpectedImprovement):
        super().__init__(dim)
        self.alpha = alpha
        self.length = 1.0
        self.length_min = 0.5**7 # need to tune
        
        self.acq_func = acq_func

    def update_state(self, next_y, model, X, y, opt_kwargs):
        super().update_state(next_y)

        # Compute the EI at incumbent before shrinking the lengthscalels
        acf = self.acq_func(model, X, prune_baseline=True)
        _, lb, ub = self.get_trust_region(model, X, y)
        bounds = torch.stack([lb, ub]).to(X.device)
        candidate, _ = optimize_acqf(
            acf, 
            bounds=bounds,
            q=1, # generate only 1 candidate instead of a batch
            **opt_kwargs
        )
        log_ei = acf(candidate).item()

        # Compute the EI at incumbent after shinking the lengthscales
        model_ast = copy.deepcopy(model)
        model_ast.covar_module.lengthscale = model_ast.covar_module.lengthscale * 0.5 # shrink the lengthscales
        acf_ast = self.acq_func(model_ast, X, prune_baseline=True)
        _, lb_ast, ub_ast = self.get_trust_region(model_ast, X, y)
        bounds_ast = torch.stack([lb_ast, ub_ast]).to(X.device)
        candidate_ast, _ = optimize_acqf(
            acf_ast, 
            bounds=bounds_ast,
            q=1, # generate only 1 candidate instead of a batch
            **opt_kwargs
        )
        log_ei_ast = acf(candidate_ast).item()

        # Compute the ratio of EI
        ratio = math.exp(log_ei_ast - log_ei)
        if ratio > self.alpha:
            self.length = max(self.length_min, self.length * 0.5)
        # elif ratio < self.alpha * 0.5:
        #     self.length = min(1.0, self.length * 2.0) # maximum length is 1.0

        print(f"EI: {math.exp(log_ei):.2f} -> {math.exp(log_ei_ast):.2f} Ratio: {ratio:.2f} Length: {self.length:.2f} Best: {self.best_value:.2f}")
        # print(model.covar_module.lengthscale)
        # print(model_ast.covar_module.lengthscale)

    def get_trust_region(self, model, X, y):
        return 0.5 * torch.ones(self.dim), torch.zeros(self.dim), torch.ones(self.dim)
    

"""
Test with AlphaRatioState using option 2.
Has bug running with cuda, need to run with cpu for experiment.
"""
class AlphaRatioStateAlter(AlphaRatioState):
    def __init__(self, dim, alpha=1.0, acq_func=qLogNoisyExpectedImprovement):
        super().__init__(dim, alpha, acq_func)
        self.length_min = 2 * 1e-4

    def update_state(self, next_y, model, X, y, opt_kwargs, covar_module):
        self.best_value = max(self.best_value, max(next_y).item())

        if self.length < self.length_min:
            self.restart_triggered = True

        acf = self.acq_func(model, X, prune_baseline=True)
        _, lb, ub = self.get_trust_region(model, X, y)
        bounds = torch.stack([lb, ub]).to(X.device)
        candidate, _ = optimize_acqf(
            acf, 
            bounds=bounds,
            q=1, # generate only 1 candidate instead of a batch
            **opt_kwargs
        )
        log_ei = acf(candidate).item()

        model_ast = copy.deepcopy(model)
        model_ast.covar_module = covar_module.to(X.device)
        mll_ast = gpytorch.mlls.ExactMarginalLogLikelihood(model_ast.likelihood, model_ast).to(X.device)
        fit_gpytorch_mll(mll_ast)
        # print("Candidate lengthscale: ", model_ast.covar_module.lengthscale)
        acf_ast = self.acq_func(model_ast, X, prune_baseline=True)
        _, lb_ast, ub_ast = self.get_trust_region(model_ast, X, y)
        bounds_ast = torch.stack([lb_ast, ub_ast]).to(X.device)
        candidate_ast, _ = optimize_acqf(
            acf_ast, 
            bounds=bounds_ast,
            q=1, # generate only 1 candidate instead of a batch
            **opt_kwargs
        )
        log_ei_ast = acf(candidate_ast).item()

        # Compute the ratio of EI
        ratio = math.exp(log_ei_ast - log_ei)
        if ratio > self.alpha:
            self.length *= 0.5
        # elif ratio < self.alpha * 0.5:
        #     self.length = min(1.0, self.length * 2.0) # maximum length is 1.0

        print(f"EI: {math.exp(log_ei):.2f} -> {math.exp(log_ei_ast):.2f} Ratio: {ratio:.2f} Length: {self.length:.2f} Best: {self.best_value:.2f}")
        # print(model.covar_module.lengthscale)
        # print(model_ast.covar_module.lengthscale)


"""
Maintain the length of the TR, success and failure counters, success and failure tolerance, etc.
In order to control the evolution of the TR length.
Modified from the BoTorch tutorial (https://botorch.org/tutorials/turbo_1).
"""
class TurboState(DummyState):
    def __init__(self, dim, success_tolerance=10, min_failure_tolerance=4.0):
        super().__init__(dim)
        self.length = 1.0
        self.length_min = 0.5**7
        self.length_max = 2.0
        self.failure_counter = 0
        self.failure_tolerance = math.ceil(max(min_failure_tolerance, float(self.dim)))
        self.success_counter = 0
        self.success_tolerance = success_tolerance

    def update_state(self, next_y):
        if max(next_y) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        # print("Successes: ", self.success_counter, " Failures: ", self.failure_counter, " Length: ", self.length)

        self.best_value = max(self.best_value, max(next_y).item())

        if self.length < self.length_min:
            self.restart_triggered = True

    """Get the trust region for the next iteration."""
    def get_trust_region(self, model, X, y):
        # Take the best observation as the center of TR
        x_center = X[y.argmax(), :].clone()

        # Rescale the TR side length according to the lengthscales in the GP model
        try:
            lengthscales = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except:
            lengthscales = model.covar_module.lengthscale.squeeze().detach()
        # lengthscales = lengthscales / lengthscales.mean()
        scale_factor = lengthscales / torch.prod(lengthscales.pow(1.0 / len(lengthscales)))

        # Define the trust region
        tr_lb = torch.clamp(x_center - scale_factor * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + scale_factor * self.length / 2.0, 0.0, 1.0)

        return x_center, tr_lb, tr_ub