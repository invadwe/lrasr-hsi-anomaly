import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import load_dataset, DATASETS
from lrasr import (global_rx, seg_rx, srd_detector, rpca_rx, lrasr_detector,
                   compute_roc_auc, hysime_mse)

EPS = np.finfo(float).eps
NAMES = ['Global-RX', 'SegRX', 'SRD', 'RPCA-RX', 'LRASR']


def run_all_detectors(X, H, W, K, P, beta, lam, seed):
    return [
        global_rx(X),
        seg_rx(X, K, seed),
        srd_detector(X, H, W, 15, 7, 5),
        rpca_rx(X),
        lrasr_detector(X, K, P, beta, lam, seed),
    ]


def plot_maps(maps, H, W, title, fname):
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
    for i in range(5):
        m = maps[i].reshape(H, W, order='F')
        ax[i].imshow(m, cmap='jet')
        ax[i].set_title(NAMES[i])
        ax[i].axis('off')
    fig.suptitle(title)
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_roc(maps, gt, title, fname, xmax=1.0):
    fig = plt.figure()
    aucs = np.zeros(5)
    for i in range(5):
        pf, pd, aucs[i] = compute_roc_auc(maps[i], gt)
        plt.plot(pf, pd, linewidth=1.5, label=NAMES[i])
    plt.xlabel('False alarm rate')
    plt.ylabel('Probability of detection')
    plt.xlim([0, xmax])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(title)
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return aucs


def false_color(cube):
    B = cube.shape[2]

    def nz(ch):
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        return np.clip((ch - lo) / (hi - lo + EPS), 0, 1)

    return np.dstack([nz(cube[:, :, int(B * 0.7)]),
                      nz(cube[:, :, int(B * 0.5)]),
                      nz(cube[:, :, int(B * 0.2)])])


def plot_comparison(cube, gt, detmap, H, W, title, fname):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].imshow(false_color(cube))
    ax[0].set_title('Imagine bruta (false color)')
    ax[0].axis('off')
    ax[1].imshow(gt.reshape(H, W, order='F'), cmap='gray')
    ax[1].set_title('Ground truth')
    ax[1].axis('off')
    im = ax[2].imshow(detmap.reshape(H, W, order='F'), cmap='jet')
    ax[2].set_title('Detectie LRASR')
    ax[2].axis('off')
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(title)
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


CONFIG = {
    'simulated': dict(fig_tag='simulat', roc_xmax=1.0, title='SIMULAT'),
    'sandiego': dict(fig_tag='sandiego', roc_xmax=1.0, title='SAN DIEGO'),
    'urban': dict(fig_tag='urban', roc_xmax=0.4, title='URBAN HYDICE'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=['all'] + list(DATASETS))
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--fig-dir', default='figures')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--K', type=int, default=15)
    ap.add_argument('--P', type=int, default=20)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    targets = list(DATASETS) if args.dataset == 'all' else [args.dataset]

    lines = ['Rezultate AUC - reproducere LRASR (Xu et al., 2016) - Python', '']
    for name in targets:
        cfg = CONFIG[name]
        tag = cfg['fig_tag']
        X, gt, H, W, cube = load_dataset(name, args.data_dir, args.seed)
        maps = run_all_detectors(X, H, W, args.K, args.P, args.beta, args.lam, args.seed)
        plot_maps(maps, H, W, 'Harti 2D - ' + cfg['title'],
                  os.path.join(args.fig_dir, 'maps_' + tag + '.png'))
        aucs = plot_roc(maps, gt, 'ROC - ' + cfg['title'],
                        os.path.join(args.fig_dir, 'roc_' + tag + '.png'),
                        xmax=cfg['roc_xmax'])
        plot_comparison(cube, gt, maps[4], H, W,
                        cfg['title'] + ': brut vs GT vs LRASR',
                        os.path.join(args.fig_dir, 'comp_' + tag + '.png'))
        header = '=== AUC (' + cfg['title'] + ') ==='
        lines.append(header)
        print(header)
        for i in range(5):
            row = '%-10s : %.4f' % (NAMES[i], aucs[i])
            lines.append(row)
            print(row)
        lines.append('')

    out = os.path.join(args.results_dir, 'rezultate_AUC.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('\nAUC salvat in', out)


if __name__ == '__main__':
    main()
