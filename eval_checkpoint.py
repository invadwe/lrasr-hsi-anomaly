import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import load_dataset
from lrasr import compute_roc_auc


def evaluate_checkpoint(ckpt_path, data_dir, fig_dir):
    ck = np.load(ckpt_path, allow_pickle=True)
    dataset = str(ck['dataset'])
    detmap = ck['detmap'].astype(float)
    H, W = int(ck['H']), int(ck['W'])
    seed = int(ck['seed'])
    _, gt, _, _, _ = load_dataset(dataset, data_dir, seed)

    pf, pd, auc = compute_roc_auc(detmap, gt)

    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, dataset)

    fig = plt.figure()
    plt.plot(pf, pd, linewidth=1.8, label='LRASR (AUC=%.4f)' % auc)
    plt.xlabel('False alarm rate')
    plt.ylabel('Probability of detection')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title('ROC din checkpoint - ' + dataset)
    fig.savefig(base + '_ckpt_roc.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(detmap.reshape(H, W, order='F'), cmap='jet')
    plt.axis('off')
    plt.title('Harta de detectie LRASR - ' + dataset)
    plt.colorbar()
    fig.savefig(base + '_ckpt_map.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    return dataset, auc, dict(K=int(ck['K']), P=int(ck['P']),
                              beta=float(ck['beta']), lam=float(ck['lam']),
                              seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='all')
    ap.add_argument('--ckpt-dir', default='checkpoints')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--fig-dir', default='figures')
    args = ap.parse_args()

    if args.checkpoint == 'all':
        paths = sorted(os.path.join(args.ckpt_dir, f)
                       for f in os.listdir(args.ckpt_dir)
                       if f.endswith('.npz'))
    else:
        paths = [args.checkpoint]

    for p in paths:
        dataset, auc, params = evaluate_checkpoint(p, args.data_dir, args.fig_dir)
        print('%-10s AUC=%.4f  params=%s' % (dataset, auc, params))


if __name__ == '__main__':
    main()
