import math
import torch

"""
A dummy state maintainer that fix the trust region length to 1.0 (hence equivalent to vanilla BO)
"""
class DummyState:
    def __init__(self, dim):
        self.dim = dim
        self.length = 1.0
        self.best_value = -float('inf')
        self.restart_triggered = False

    def update_state(self, next_y):
        self.best_value = max(self.best_value, max(next_y).item())

    def get_trust_region(self, model, X, y):
        return 0.5 * torch.ones(self.dim), torch.zeros(self.dim), torch.ones(self.dim)

"""
Maintain the length of the TR, success and failure counters, success and failure tolerance, etc.
In order to control the evolution of the TR length.
Modified from the BoTorch tutorial (https://botorch.org/tutorials/turbo_1).
"""
class TurboState(DummyState):
    def __init__(self, dim, success_tolerance=10, min_failure_tolerance=4.0):
        super().__init__(dim)
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
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

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

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