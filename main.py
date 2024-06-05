from omegaconf import DictConfig
from os.path import dirname, join
import pandas as pd
import numpy as np
import math
import os
import hydra
import torch

from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS
)
from benchmarking.gp_priors import (
    MODELS,
    get_covar_module
)

from state_evol import (
    DummyState, 
    SchedulerState,
    TurboState, 
    AlphaRatioState,
)
# from turbo import VanillaTuRBO
from ls_evol import BaseLengthEvol
from soft_winsorization import SigmoidBO

get_evol_state_maintainer = {
    'dummy': DummyState,
    'scheduler': SchedulerState,
    'turbo': TurboState,
    'AR': AlphaRatioState,
}        

@hydra.main(config_path='./configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the bounds of the input space
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0).to(device)

    # Get the model and model_kwargs
    model = MODELS[cfg.model.gp]
    covar_model_params = {
        "model_name": cfg.model.model_name,
        "dims": len(bounds.T),
        "gp_params": cfg.model.get('gp_params', None),
        "gp_constraints": cfg.model.get('gp_constraints', {})
    }

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
    # optimizer = VanillaTuRBO(
    #     model=model,
    #     model_params=covar_model_params,
    #     acq_func=ACQUISITION_FUNCTIONS[cfg.acq.acq_func],
    #     opt_kwargs=dict(cfg.acq_opt),
    #     objective=test_function,
    #     bounds=bounds,
    #     num_init=num_init,
    #     num_bo=num_bo,
    #     num_turbo=num_turbo,
    #     device=device,
    #     seed=cfg.seed,
    #     gss=cfg.gss if hasattr(cfg, 'gss') else None
    # )
    # Remark. If using AlphaRatioStateAlter, let device='cpu'.
    if cfg.evol_state_maintainer == 'sigmoid':
        # Remark. If using SigmoidBO, let device='cpu'.
        optimizer = SigmoidBO(
            model=model,
            model_params=covar_model_params,
            acq_func=ACQUISITION_FUNCTIONS[cfg.acq.acq_func],
            opt_kwargs=dict(cfg.acq_opt),
            objective=test_function,
            bounds=bounds,
            num_init=num_init,
            num_bo=num_bo,
            device='cpu',
            seed=cfg.seed,
        )
    else:
        optimizer = BaseLengthEvol(
            model=model,
            model_params=covar_model_params,
            acq_func=ACQUISITION_FUNCTIONS[cfg.acq.acq_func],
            opt_kwargs=dict(cfg.acq_opt),
            objective=test_function,
            bounds=bounds,
            num_init=num_init,
            num_bo=num_bo,
            device=device,
            seed=cfg.seed,
            evol_state_maintainer=get_evol_state_maintainer[cfg.evol_state_maintainer],
        )

    # run the optimization
    optimizer.run(
        save_path=join(dirname(os.path.abspath(__file__)), cfg.result_path),
        save_name=cfg.experiment_name
    )

if __name__ == '__main__':
    main()
