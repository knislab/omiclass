"""Process-local reproducibility scopes; never change another worker's state."""
from contextlib import contextmanager
import os
import random

import numpy as np


@contextmanager
def random_state_scope(seed=None, *, torch_enabled=True, threads=None,
                       deterministic=None, cudnn_benchmark=None, cublas=False, cuda=False):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    old_cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    torch = None
    cpu_state = cuda_states = None
    settings_saved = False
    try:
        if cublas:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if torch_enabled:
            import torch
            cpu_state = torch.random.get_rng_state()
            cuda_states = (torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                           and (cuda or torch.cuda.is_initialized()) else None)
            old_threads = torch.get_num_threads()
            old_deterministic = torch.are_deterministic_algorithms_enabled()
            old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
            cudnn_flags = {name: getattr(torch.backends.cudnn, name)
                           for name in ('benchmark', 'deterministic', 'allow_tf32')}
            old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            settings_saved = True
            if threads is not None:
                torch.set_num_threads(int(threads))
            if deterministic is not None:
                torch.use_deterministic_algorithms(bool(deterministic))
            if cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            if torch is not None:
                torch.random.default_generator.manual_seed(int(seed))
                if cuda_states is not None:
                    torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and cpu_state is not None:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        if settings_saved:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
            for name, value in cudnn_flags.items():
                setattr(torch.backends.cudnn, name, value)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        if old_cublas is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = old_cublas
