from typing import TypedDict

from typing import NotRequired, TypedDict

import torch
import torchaudio


class Sample(TypedDict):
    wav: torch.Tensor
    sample_rate: int
    label: list
    tokens: list


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


def apply_gain(sample: Sample, gain_db: float) -> Sample:
    """
    Inplace operation.
    """

    sample['wav'] = torchaudio.functional.gain(sample['wav'], gain_db)

    return sample