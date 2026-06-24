import os
import argparse
import time
import numpy as np
from data import load_dataset, DATASETS
from lrasr import lrasr_detector, compute_roc_auc


def train_one(dataset, data_dir, out_dir, K, P, beta, lam, seed):
    X, gt, H, W, cube = load_dataset(dataset, data_dir, seed)
    t0 = time.time()
    detmap, D, S, E = lrasr_detector(X, K, P, beta, lam, seed, return_parts=True)
    elapsed = time.time() - t0
    _, _, auc = compute_roc_auc(detmap, gt)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, dataset + '.npz')
    np.savez_compressed(
        path,
        dataset=dataset,
        K=K, P=P, beta=beta, lam=lam, seed=seed,
        H=H, W=W,
        detmap=detmap.astype(np.float32),
        D=D.astype(np.float32),
        auc=auc,
        elapsed=elapsed,
    )
    return path, auc, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=['all'] + list(DATASETS))
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='checkpoints')
    ap.add_argument('--K', type=int, default=15)
    ap.add_argument('--P', type=int, default=20)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    targets = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    for name in targets:
        path, auc, elapsed = train_one(
            name, args.data_dir, args.out_dir,
            args.K, args.P, args.beta, args.lam, args.seed)
        print('%-10s AUC=%.4f  timp=%6.1fs  ->  %s'
              % (name, auc, elapsed, path))


if __name__ == '__main__':
    main()
