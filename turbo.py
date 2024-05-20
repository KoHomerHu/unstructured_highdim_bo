import math
import torch
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling

"""
Maintain the length of the TR, success and failure counters, success and failure tolerance, etc.
In order to control the evolution of the TR length.
Modified from the BoTorch tutorial (https://botorch.org/tutorials/turbo_1).
"""
class TurboState:
    def __init__(self, dim, success_tolerance=10):
        self.dim = dim
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = math.ceil(max(4.0, float(self.dim)))
        self.success_counter = 0
        self.success_tolerance = success_tolerance
        self.best_value = float('inf')
        self.restart_triggered = False

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
    

"""
Given the current state and a GP model with data X and Y
Use Thompson sampling to generate the next point in the trust region.
Modified from the BoTorch tutorial (https://botorch.org/tutorials/turbo_1).
"""
def generate_next_point(state, model, X, y):
    # Take the best observation as the center of TR
    x_center = X[y.argmax(), :].clone()

    # Rescale the TR side length according to the lengthscales in the GP model
    lengthscales = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    lengthscales = lengthscales / lengthscales.mean()
    scale_factor = lengthscales / torch.prod(lengthscales.pow(1.0 / len(lengthscales)))

    # Define the trust region
    tr_lb = torch.clamp(x_center - scale_factor * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + scale_factor * state.length / 2.0, 0.0, 1.0)

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