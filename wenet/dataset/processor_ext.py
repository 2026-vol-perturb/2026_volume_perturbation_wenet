import random
from typing import TypedDict, Union
import math

import torch
import torchaudio


class Sample(TypedDict):
    wav: torch.Tensor
    sample_rate: int

def normalize_power(sample: Sample):
    """
    Inplace operation.
    """
    x = sample['wav']

    energy = torch.sum(x ** 2)
    power = energy / x.size(-1)
    x /= torch.sqrt(power)

    sample['wav'] = x

    return sample


def apply_random_gain(sample: Sample, db: list[float], interval_s: float = None, vary_db: list[float] = None, uniform: float = False, mode: Union['dB', 'amplitude'] = 'dB') -> Sample:
    """
    This function combines various forms of volume perturbation and block-wise volume perturbation.
    For parameter usage, see the config files in the recipes.

    Notice: The `db` and 'vary_db' parameter names became misleading:
    - In mode = 'dB', they are interpreted as decibel values, as their names imply.
    - In mode = 'amplitude', they are interpreted as amplitude factors, contrary to their names!

    Inplace operation.
    """

    def random_gain(array):
        if uniform:
            val = random.uniform(array[0], array[1])
        else:
            val = random.choice(array)

        if mode == 'amplitude':
            assert val > 0, 'amplitude factor must be > 0'
            gain_db = 20 * math.log10(val)
        else:
            gain_db = val

        return gain_db

    sample['wav'] = torchaudio.functional.gain(sample['wav'], random_gain(db))

    if interval_s is not None or vary_db is not None:
        length = sample['wav'].size(-1)

        sample_rate = sample['sample_rate']
        interval = int(interval_s * sample_rate)

        # Prevent change points being in the same spot.
        offset = -random.randint(0, interval)

        for i in range(offset, length, interval):
            sample['wav'][:, max(i,0):i+interval] = torchaudio.functional.gain(sample['wav'][:, max(i,0):i+interval], random_gain(vary_db))

    return sample
