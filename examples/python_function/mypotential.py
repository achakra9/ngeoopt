"""Example energy functions for the 'function' backend."""

import numpy as np


def morse(symbols, coords, De=0.17, a=1.0, re=1.8):
    """Pairwise Morse potential. coords in Bohr, energy in Hartree."""
    e = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            r = np.linalg.norm(coords[i] - coords[j])
            e += De * (1 - np.exp(-a * (r - re))) ** 2 - De
    return e


def torch_model(symbols, coords, device="cpu"):
    """Same idea in torch: `device` is 'mps' on Apple silicon, 'cuda:0' on NVIDIA."""
    import torch
    x = torch.as_tensor(coords, dtype=torch.float32, device=device)
    r = torch.cdist(x, x)
    iu = torch.triu_indices(len(x), len(x), 1, device=device)
    r = r[iu[0], iu[1]]
    return (0.17 * (1 - torch.exp(-(r - 1.8))) ** 2 - 0.17).sum()
