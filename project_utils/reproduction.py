"""Portable I/O helpers; no changes to SelEx objectives or clustering."""
from pathlib import Path
import torch


def load_dino_backbone(path):
    state = torch.load(path, map_location='cpu')
    if 'teacher' in state:
        state = state['teacher']
        state = {k.removeprefix('module.').removeprefix('backbone.'): v
                 for k, v in state.items() if 'head.' not in k}
    return state


def limited_batches(loader, limit=0):
    if limit < 0:
        raise ValueError('Batch limit must be nonnegative')
    from itertools import islice
    return islice(loader, limit) if limit else iter(loader)


def save_training_state(path, model, head, optimizer, scheduler, epoch, **extra):
    import numpy as np
    import random
    torch.save(dict(model=model.state_dict(), head=head.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    epoch=epoch, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    **extra), path)


def restore_training_state(path, model, head, optimizer, scheduler):
    import numpy as np
    import random
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])
    head.load_state_dict(state['head'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
