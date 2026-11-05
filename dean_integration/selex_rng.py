"""Private Python/NumPy/Torch RNG with restoration even if augmentation fails."""
from contextlib import contextmanager
import random

import numpy as np
import torch


class SelExRNG:
    def __init__(self, seed):
        self.seed = (int(seed) ^ 0x5E1E) % 2**32
        self.python = random.Random(self.seed).getstate()
        self.numpy = np.random.RandomState(self.seed).get_state()
        self.cpu = torch.Generator().manual_seed(self.seed).get_state()
        self.cuda = {}

    @contextmanager
    def scope(self, device="cpu"):
        device = torch.device(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        python, numpy = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=devices):
            try:
                random.setstate(self.python)
                np.random.set_state(self.numpy)
                torch.set_rng_state(self.cpu)
                for index in devices:
                    if index not in self.cuda:
                        self.cuda[index] = torch.Generator(device=f"cuda:{index}").manual_seed(self.seed).get_state()
                    torch.cuda.set_rng_state(self.cuda[index], index)
                yield
            finally:
                self.python = random.getstate()
                self.numpy = np.random.get_state()
                self.cpu = torch.get_rng_state()
                for index in devices:
                    self.cuda[index] = torch.cuda.get_rng_state(index)
                random.setstate(python)
                np.random.set_state(numpy)
