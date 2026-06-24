import numpy as np
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps


def svt(Y, tau):
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    r = int(np.sum(s > 0))
    if r == 0:
        return np.zeros_like(Y)
    return (U[:, :r] * s[:r]) @ Vt[:r, :]


def soft_threshold(Y, tau):
    return np.sign(Y) * np.maximum(np.abs(Y) - tau, 0.0)


def solve_l21(Y, tau):
    nrm = np.sqrt(np.sum(Y ** 2, axis=0))
    scale = np.maximum(1.0 - tau / (nrm + EPS), 0.0)
    scale[nrm <= tau] = 0.0
    return Y * scale[np.newaxis, :]


def solve_lrasr_ladmap(X, D, beta, lam, max_iter=500):
    N = X.shape[1]
    m = D.shape[1]
    S = np.zeros((m, N))
    J = np.zeros((m, N))
    E = np.zeros_like(X)
    Y1 = np.zeros_like(X)
    Y2 = np.zeros((m, N))
    mu, mu_max, rho0 = 0.01, 1e10, 1.1
    eps1, eps2 = 1e-6, 1e-2
    eta1 = np.linalg.norm(D, 2) ** 2
    normX = np.linalg.norm(X, 'fro')
    DT = D.T

    for _ in range(max_iter):
        S_old, J_old, E_old = S.copy(), J.copy(), E.copy()
        grad = DT @ (X - D @ S - E + Y1 / mu) - (S - J + Y2 / mu)
        S = svt(S + grad / eta1, 1.0 / (eta1 * mu))
        J = soft_threshold(S + Y2 / mu, beta / mu)
        E = solve_l21(X - D @ S + Y1 / mu, lam / mu)
        Y1 = Y1 + mu * (X - D @ S - E)
        Y2 = Y2 + mu * (S - J)
        leq1 = np.linalg.norm(X - D @ S - E, 'fro') / normX
        rel_chg = mu * max(np.sqrt(eta1) * np.linalg.norm(S - S_old, 'fro'),
                           np.linalg.norm(J - J_old, 'fro'),
                           np.linalg.norm(E - E_old, 'fro')) / normX
        rho = rho0 if rel_chg <= eps2 else 1.0
        mu = min(mu_max, rho * mu)
        if leq1 < eps1 and rel_chg < eps2:
            break
    return S, E


def construct_dictionary(X, K, P, seed=0):
    B, N = X.shape
    Xt = X.T
    idx = KMeans(n_clusters=K, n_init=3, max_iter=200,
                 random_state=seed).fit_predict(Xt)
    cols = []
    for i in range(K):
        members = np.where(idx == i)[0]
        if members.size < P:
            continue
        Xi = Xt[members, :]
        mu = Xi.mean(axis=0)
        C = np.cov(Xi, rowvar=False) + 1e-6 * np.eye(B)
        diff = Xi - mu
        PD = np.sum((diff @ np.linalg.inv(C)) * diff, axis=1)
        order = np.argsort(PD)
        chosen = members[order[:P]]
        cols.append(X[:, chosen])
    if not cols:
        raise RuntimeError('Dictionar gol: scade P sau verifica datele.')
    return np.hstack(cols)


def lrasr_detector(X, K, P, beta, lam, seed=0, return_parts=False):
    D = construct_dictionary(X, K, P, seed)
    S, E = solve_lrasr_ladmap(X, D, beta, lam)
    detmap = np.sqrt(np.sum(E ** 2, axis=0))
    if return_parts:
        return detmap, D, S, E
    return detmap


def global_rx(X):
    B, N = X.shape
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    C = (Xc @ Xc.T) / N + 1e-6 * np.eye(B)
    Cinv = np.linalg.inv(C)
    return np.sum((Cinv @ Xc) * Xc, axis=0)


def seg_rx(X, K, seed=0):
    B, N = X.shape
    idx = KMeans(n_clusters=K, n_init=3, max_iter=200,
                 random_state=seed).fit_predict(X.T)
    detmap = np.zeros(N)
    for i in range(K):
        members = np.where(idx == i)[0]
        if members.size < 2:
            continue
        Xi = X[:, members]
        mu = Xi.mean(axis=1, keepdims=True)
        C = np.cov(Xi, rowvar=True) + 1e-6 * np.eye(B)
        Xc = Xi - mu
        scores = np.sum((np.linalg.solve(C, Xc)) * Xc, axis=0)
        s_mean, s_std = scores.mean(), scores.std() + EPS
        detmap[members] = (scores - s_mean) / s_std
    return detmap


def rpca_rx(X, lam=None):
    B, N = X.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(B, N))
    normX = np.linalg.norm(X, 'fro')
    Y = X / max(np.linalg.norm(X, 2), np.max(np.abs(X)) / lam)
    mu = 1.25 / np.linalg.norm(X, 2)
    mu_max, rho, tol = mu * 1e7, 1.5, 1e-7
    L = np.zeros((B, N))
    Sp = np.zeros((B, N))
    for _ in range(500):
        U, s, Vt = np.linalg.svd(X - Sp + Y / mu, full_matrices=False)
        s2 = np.maximum(s - 1 / mu, 0)
        r = int(np.sum(s2 > 0))
        L = (U[:, :r] * s2[:r]) @ Vt[:r, :]
        T = X - L + Y / mu
        Sp = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0)
        Z = X - L - Sp
        Y = Y + mu * Z
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(Z, 'fro') / normX < tol:
            break
    return np.sqrt(np.sum(Sp ** 2, axis=0))


def _omp(D, y, K0):
    m = D.shape[1]
    nrm = np.sqrt(np.sum(D ** 2, axis=0))
    nrm[nrm == 0] = 1
    Dn = D / nrm
    idxset = []
    x = np.zeros(m)
    residual = y.copy()
    for _ in range(min(K0, m)):
        j = int(np.argmax(np.abs(Dn.T @ residual)))
        if j in idxset:
            break
        idxset.append(j)
        Dsel = Dn[:, idxset]
        xs, *_ = np.linalg.lstsq(Dsel, y, rcond=None)
        residual = y - Dsel @ xs
        if np.linalg.norm(residual) < 1e-6:
            break
    if idxset:
        xs, *_ = np.linalg.lstsq(Dn[:, idxset], y, rcond=None)
        x[idxset] = xs / nrm[idxset]
    return x


def srd_detector(X, nrows, ncols, w_out, w_in, K0=5):
    B, N = X.shape
    detmap = np.zeros(N)
    ho, hi = w_out // 2, w_in // 2
    for c in range(ncols):
        for r in range(nrows):
            idx = r + c * nrows
            y = X[:, idx]
            atoms = []
            for cc in range(max(0, c - ho), min(ncols, c + ho + 1)):
                for rr in range(max(0, r - ho), min(nrows, r + ho + 1)):
                    if abs(rr - r) <= hi and abs(cc - c) <= hi:
                        continue
                    atoms.append(rr + cc * nrows)
            if not atoms:
                detmap[idx] = 0.0
                continue
            Dloc = X[:, atoms]
            coef = _omp(Dloc, y, K0)
            detmap[idx] = np.linalg.norm(y - Dloc @ coef)
    return detmap


def implant_targets(cube, t_spectrum):
    H, W, B = cube.shape
    rows_pos = np.round(np.linspace(0.2, 0.8, 4) * H).astype(int)
    cols_pos = np.round(np.linspace(0.2, 0.8, 4) * W).astype(int)
    rows_pos = np.clip(rows_pos - 1, 0, H - 1)
    cols_pos = np.clip(cols_pos - 1, 0, W - 1)
    f_per_row = [0.05, 0.1, 0.2, 0.4]
    cube_out = cube.copy()
    gt = np.zeros((H, W))
    t = t_spectrum.ravel()
    for ri in range(4):
        f = f_per_row[ri]
        r = rows_pos[ri]
        for ci in range(4):
            c = cols_pos[ci]
            b = cube[r, c, :]
            cube_out[r, c, :] = f * t + (1 - f) * b
            gt[r, c] = 1
    return cube_out, gt


def compute_roc_auc(detmap, gt, n_thr=200):
    d = detmap.ravel(order='F')
    g = gt.ravel(order='F') > 0
    nT, nB = g.sum(), (~g).sum()
    thr = np.linspace(d.max(), d.min(), n_thr)
    pd = np.zeros(n_thr)
    pf = np.zeros(n_thr)
    for i in range(n_thr):
        det = d >= thr[i]
        pd[i] = np.sum(det & g) / max(nT, 1)
        pf[i] = np.sum(det & ~g) / max(nB, 1)
    order = np.argsort(pf)
    pf, pd = pf[order], pd[order]
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(pd, pf)
    else:
        auc = np.trapz(pd, pf)
    return pf, pd, auc


def hysime_mse(X):
    B, N = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    R = (Xc @ Xc.T) / N
    w, V = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    Ev = V[:, order]
    mse = np.zeros(B)
    for k in range(1, B + 1):
        Ek = Ev[:, :k]
        Xrec = Ek @ (Ek.T @ Xc)
        err = Xc - Xrec
        mse[k - 1] = np.sum(err ** 2) / (N * B)
    return mse
