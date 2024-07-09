from botorch.test_functions import (
    Hartmann,
    Levy,
    Ackley,
    Michalewicz,
    EggHolder,
    Griewank
)
from benchmarking.mujoco_task import MujocoFunction
from ax.modelbridge.registry import Models
from benchmarking.synthetic import (
    Embedded
)
from botorch.acquisition import (
    qNoisyExpectedImprovement,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement


def get_test_function(name: str, noise_std: float, seed: int = 0,bounds=None):
    
    TEST_FUNCTIONS = {
        'levy4_4': (Embedded, dict(function=Levy(dim=4), noise_std=noise_std, negate=True, dim=4)),
        'levy4_25': (Embedded, dict(function=Levy(dim=4), noise_std=noise_std, negate=True, dim=25)),
        'levy4_100': (Embedded, dict(function=Levy(dim=4), noise_std=noise_std, negate=True, dim=100)),
        'levy4_300': (Embedded, dict(function=Levy(dim=4), noise_std=noise_std, negate=True, dim=300)),
        'levy4_1000': (Embedded, dict(function=Levy(dim=4), noise_std=noise_std, negate=True, dim=1000)),
        'hartmann6_6': (Embedded, dict(function=Hartmann(dim=6), noise_std=noise_std, negate=True, dim=6)),
        'hartmann6_25': (Embedded, dict(function=Hartmann(dim=6), noise_std=noise_std, negate=True, dim=25)),
        'hartmann6_100': (Embedded, dict(function=Hartmann(dim=6), noise_std=noise_std, negate=True, dim=100)),
        'hartmann6_300': (Embedded, dict(function=Hartmann(dim=6), noise_std=noise_std, negate=True, dim=300)),
        'hartmann6_1000': (Embedded, dict(function=Hartmann(dim=6), noise_std=noise_std, negate=True, dim=1000)),
        'ackley4_25': (Embedded, dict(function=Ackley(dim=4), noise_std=noise_std, negate=True, dim=25)),
        'ackley4_100': (Embedded, dict(function=Ackley(dim=4), noise_std=noise_std, negate=True, dim=100)),
        'michalewicz10_25': (Embedded, dict(function=Michalewicz(dim=10), noise_std=noise_std, negate=True, dim=25)),
        'michalewicz10_100': (Embedded, dict(function=Michalewicz(dim=10), noise_std=noise_std, negate=True, dim=100)),
        'eggholder2_25': (Embedded, dict(function=EggHolder(), noise_std=noise_std, negate=True, dim=25)),
        'griewank6_25': (Embedded, dict(function=Griewank(dim=6), noise_std=noise_std, negate=True, dim=25)),
   }

    if name in TEST_FUNCTIONS.keys():
        function = TEST_FUNCTIONS[name]
        
    elif name == 'lasso_dna':
        from benchmarking.lassobench_task import LassoRealFunction
        function = LassoRealFunction, dict(negate=True, seed=seed, pick_data='dna')
    elif name == 'mopta':
        from benchmarking.benchsuite_task import BenchSuiteFunction
        function = (BenchSuiteFunction, dict(negate=True, task_id='mopta'))
    elif name == 'svm':
        from benchmarking.benchsuite_task import BenchSuiteFunction
        function = (BenchSuiteFunction, dict(negate=True, task_id='svm'))
    elif name == 'swimmer':
        from benchmarking.mujoco_task import MujocoFunction
        function = (MujocoFunction, dict(negate=True, bounds=bounds, container='mujoco', task_id='swimmer'))
    elif name == 'ant':
        from benchmarking.mujoco_task import MujocoFunction
        function = (MujocoFunction, dict(negate=True, bounds=bounds, container='mujoco', task_id='ant'))
    elif name == 'humanoid':
        from benchmarking.mujoco_task import MujocoFunction
        function = (MujocoFunction, dict(negate=True, bounds=bounds, container='mujoco', task_id='humanoid'))
    elif name == 'lunar_lander':
        from benchmarking.lunar_lander_task import LunarLanderFunction
        function = (LunarLanderFunction, dict(negate=False))
    elif name == 'lunar_lander_continuous':
        from benchmarking.lunar_lander_continuous_task import LunarLanderContinuousFunction
        function = (LunarLanderContinuousFunction, dict(negate=False))
    elif name == 'cartpole':
        from benchmarking.cartpole_task import CartPoleFunction
        function = (CartPoleFunction, dict(negate=False))
    elif name == 'bipedal_walker':
        from benchmarking.bipedal_walker_task import BipedalWalkerFunction
        function = (BipedalWalkerFunction, dict(negate=False))
    elif name == 'walker2d':
        from benchmarking.walker2d_task import Walker2DFunction
        function = (Walker2DFunction, dict(negate=False))
    else:
        raise ValueError(f"Function {name} is not available - feel free to add it!")    
    
    function_init = function[0](**function[1])
    return function_init


ACQUISITION_FUNCTIONS = {
    'NEI': qNoisyExpectedImprovement,
    'qLogNEI': qLogNoisyExpectedImprovement,
}


INITS = {
    'sobol': Models.SOBOL,
}
