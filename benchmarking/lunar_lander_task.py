from botorch.test_functions.base import BaseTestProblem
import numpy as np
from torch import Tensor
import gymnasium as gym


def heuristic_controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


def trial(weights, seed):
    env = gym.make('LunarLander-v2')
    np.random.seed(seed)
    env.action_space.seed(seed)
    s, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    while True:
        a = heuristic_controller(s, weights)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += r
        steps += 1
        if done:
            break
    return total_reward


class LunarLanderFunction(BaseTestProblem):
    def __init__(self, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.seed = seed
        self.dim = 12
        self._bounds = [(0.0, 2.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)
        
    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        X_np = X.detach().numpy().flatten().astype(np.float64)
        
        results = [trial(X_np, seed+179*i) for i in range(1)]
        val = np.mean(results)

        return Tensor([val])