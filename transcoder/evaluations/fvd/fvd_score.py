import numpy as np
import os
import scipy
import torch
from typing import Callable, Tuple

from transcoder.evaluations.fvd.utils import open_url


def get_i3d_model():
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(dirname, 'i3d_torchscript.pt')):
        with open_url(detector_url, verbose=False) as f:
            detector = torch.jit.load(f).eval()
        detector.save(os.path.join(dirname, 'i3d_torchscript.pt'))
    else:
        detector = torch.jit.load(os.path.join(dirname, 'i3d_torchscript.pt')).eval()
    return detector


def init_torch_model(device: str='cpu') -> Callable:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    return lambda x: detector(torch.from_numpy(x), **detector_kwargs).numpy()


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma


def compute_fvd_score(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))

    return float(fid)
