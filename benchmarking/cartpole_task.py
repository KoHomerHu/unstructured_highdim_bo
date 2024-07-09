from botorch.test_functions.base import BaseTestProblem
import numpy as np
from torch import Tensor
import gymnasium as gym


class PID:
    def __init__(self, kp, kd, ki, goal):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.goal = goal
        self.prev_error = 0
        self.integral = 0

    def observe(self, state):
        error = self.goal - state
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.kd * derivative + self.ki * self.integral


class Controller:
    def __init__(self, w):
        self.cart = PID(kp=w[0], kd=w[1], ki=0, goal=0)
        self.pole = PID(kp=w[2], kd=w[3], ki=0, goal=0)

    def observe(self, cart_position, pole_angle):
        u_cart = self.cart.observe(cart_position)
        u_pole = self.pole.observe(pole_angle)
        action = 1 if u_pole + u_cart < 0 else 0
        return action


def trial(weights, seed):
    controller = Controller(weights)

    env = gym.make('CartPole-v1')
    np.random.seed(seed)
    env.action_space.seed(seed)
    s, _ = env.reset(seed=seed, options={"low":-0.05, "high":0.05})
    total_reward = 0
    steps = 0
    while True:
        a = controller.observe(s[0], s[2])
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += r
        steps += 1
        if done:
            break
    return total_reward


class CartPoleFunction(BaseTestProblem):
    def __init__(self, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.seed = seed
        self.dim = 4
        self._bounds = [(0.0, 100.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)
        
    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        X_np = X.detach().numpy().flatten().astype(np.float64)
        
        results = [trial(X_np, seed+179*i) for i in range(10)]
        val = np.mean(results)

        return Tensor([val])