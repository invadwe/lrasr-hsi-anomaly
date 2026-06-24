import os
import numpy as np
from scipy.io import loadmat
from lrasr import implant_targets

EPS = np.finfo(float).eps

BAD_BANDS_SANDIEGO = (list(range(1, 7)) + list(range(33, 36)) +
                      list(range(94, 98)) + list(range(107, 114)) +
                      list(range(153, 167)) + list(range(221, 225)))


def _good_bands_sandiego():
    bad = [b - 1 for b in BAD_BANDS_SANDIEGO]
    return np.setdiff1d(np.arange(224), bad)


def load_simulated(data_dir, seed=0):
    good = _good_bands_sandiego()
    SD = loadmat(os.path.join(data_dir, 'Sandiego.mat'))['Sandiego'].astype(float)
    cubeG = SD[:, :, good]
    Rs, Cs, _ = cubeG.shape
    win = 100
    r0, c0 = 280, 20
    r0 = min(r0, Rs - win)
    c0 = min(c0, Cs - win)
    sub = SD[r0:r0 + win, c0:c0 + win, :][:, :, good]
    H, W, B = sub.shape
    bg_mean = sub.reshape(H * W, B, order='F').mean(axis=0)
    allpx = cubeG.reshape(Rs * Cs, B, order='F')
    cosang = (allpx @ bg_mean) / (np.sqrt(np.sum(allpx ** 2, axis=1)) *
                                  np.linalg.norm(bg_mean) + EPS)
    ang_all = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    order = np.argsort(ang_all)[::-1]
    j = order[int(round(0.05 * len(order)))]
    t = allpx[j, :]
    sub_imp, gt = implant_targets(sub, t)
    X = sub_imp.reshape(H * W, B, order='F').T
    X = X / X.max()
    return X, gt, H, W, sub_imp


def load_sandiego(data_dir, seed=0):
    good = _good_bands_sandiego()
    SD = loadmat(os.path.join(data_dir, 'Sandiego.mat'))['Sandiego'].astype(float)
    PG = loadmat(os.path.join(data_dir, 'PlaneGT.mat'))['PlaneGT'].astype(float)
    sub = SD[0:100, 0:100, :][:, :, good]
    H, W, B = sub.shape
    X = sub.reshape(H * W, B, order='F').T
    X = X / X.max()
    return X, PG, H, W, sub


def load_urban(data_dir, seed=0):
    HY = loadmat(os.path.join(data_dir, 'HYDICE_urban.mat'))
    cube = HY['data'].astype(float)
    gt = HY['map'].astype(float)
    H, W, B = cube.shape
    X = cube.reshape(H * W, B, order='F').T
    X = X / X.max()
    return X, gt, H, W, cube


DATASETS = {
    'simulated': load_simulated,
    'sandiego': load_sandiego,
    'urban': load_urban,
}


def load_dataset(name, data_dir, seed=0):
    if name not in DATASETS:
        raise ValueError('Dataset necunoscut: ' + name +
                         '. Optiuni: ' + ', '.join(DATASETS))
    return DATASETS[name](data_dir, seed)
