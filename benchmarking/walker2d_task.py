from botorch.test_functions.base import BaseTestProblem
import numpy as np
from torch import Tensor
import gymnasium as gym


class Normalizer():
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def trial(weights, seed):
    normalizer = Normalizer(17)
    env = gym.make('Walker2d-v3')
    np.random.seed(seed)
    env.action_space.seed(seed)
    s, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    while True:
        normalizer.observe(s)
        s = normalizer.normalize(s)
        a = weights.dot(s)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += r
        steps += 1
        if done:
            break
    return total_reward


class Walker2DFunction(BaseTestProblem):
    def __init__(self, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.seed = seed
        self.dim = 102
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)
        
    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        X_np = X.detach().numpy().astype(np.float64)
        X_np = X_np.reshape(6, 17)
        
        results = [trial(X_np, seed+179*i) for i in range(1)]
        val = np.mean(results)

        return Tensor([val])