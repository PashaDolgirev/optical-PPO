"""Small analysis helpers shared by the characterization scripts."""

import itertools
import numpy as np
import torch


@torch.no_grad()
def lyapunov(solver, a, F0, f, T=100.0, tau=1.0, d0=1e-7, seed=1):
    """
    Largest Lyapunov exponent (Benettin): evolve each trajectory next to a copy displaced
    by d0, renormalise the displacement every tau, average the log growth. Returns (B,)
    exponents in units of the LLE time (1 / (kappa/2)). Use a complex128 solver.

    a : (B, N) states on the attractor,  f : (B, n_drive) sub-band amplitudes held fixed.
    """
    B = a.shape[0]
    g = torch.Generator().manual_seed(seed)
    pert = torch.view_as_complex(torch.randn(*a.shape, 2, dtype=solver.rdtype, generator=g)).to(solver.dtype)
    pert = pert / pert.abs().pow(2).sum(-1, keepdim=True).sqrt() * d0
    solver.set_drive(F0, torch.cat([f, f]))                    # reference and displaced copies share the drive
    ab = torch.cat([a, a + pert])
    n, log_growth = int(round(tau / solver.dt)), torch.zeros(B, dtype=torch.float64)
    for _ in range(int(round(T / tau))):
        ab, _ = solver.evolve(ab, n)
        ref, dis = ab[:B], ab[B:]
        d = (dis - ref).abs().pow(2).sum(-1).sqrt()
        log_growth += torch.log(d / d0).double()
        ab = torch.cat([ref, ref + (dis - ref) * (d0 / d)[:, None]])
    return (log_growth / T).numpy()


def split_half(S1, S2):
    """
    S1, S2 : (n, n_features) features from two INDEPENDENT chaotic realisations driven by the
    same n inputs. Returns per-feature (signal_var, noise_var):
        noise_var  = Var[S | input]            = 1/2 <(S1 - S2)^2>
        signal_var = Var_input( E[S | input] ) = Cov(S1, S2)
    """
    noise = 0.5 * ((S1 - S2) ** 2).mean(0)
    signal = ((S1 - S1.mean(0)) * (S2 - S2.mean(0))).mean(0)
    return signal, noise


def ridge_decode_r2(X, Y, lams=np.logspace(-6, 2, 17), train_frac=0.7):
    """
    How well can a LINEAR readout of features X (n, d) reconstruct targets Y (n, k)?
    Standardise X, ridge-fit on the first train_frac of the rows, report R^2 per target
    on the rest at the best lambda (selected on that same held-out part: a ceiling estimate).
    """
    n = len(X); ntr = int(train_frac * n)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-12
    Xs = (X - mu) / sd
    Xtr, Xte, Ytr, Yte = Xs[:ntr], Xs[ntr:], Y[:ntr], Y[ntr:]
    ym, best = Ytr.mean(0), None
    for lam in lams:
        W = np.linalg.solve(Xtr.T @ Xtr + lam * ntr * np.eye(X.shape[1]), Xtr.T @ (Ytr - ym))
        r2 = 1 - ((Yte - (Xte @ W + ym)) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
        if best is None or r2.mean() > best.mean():
            best = r2
    return best


def poly_design(Y, kind, degree=4):
    """
    Design matrices for the variance decomposition of a feature S(s~), s~ in [-1, 1]^d:
      "linear"   : 1, s_j
      "additive" : 1, s_j^p (p <= degree)                   -- no mixing between inputs
      "pairwise" : additive + s_i^p s_j^q (p, q in {1, 2})  -- two-input mixing
    """
    d = Y.shape[1]
    cols = [np.ones(len(Y))]
    if kind == "linear":
        cols += [Y[:, j] for j in range(d)]
    else:
        for j in range(d):
            cols += [Y[:, j] ** p for p in range(1, degree + 1)]
        if kind == "pairwise":
            for i, j in itertools.combinations(range(d), 2):
                cols += [Y[:, i] ** p * Y[:, j] ** q for p in (1, 2) for q in (1, 2)]
    return np.stack(cols, 1)


def explained_signal_fraction(Y, S1, S2, kind):
    """
    Fraction of the SIGNAL variance of each feature captured by the model class `kind`.
    Fit on realisation 1 / first half of the inputs, score on realisation 2 / second half,
    subtract the (known) noise floor so that a perfect model scores 1 regardless of noise.
    """
    signal, noise = split_half(S1, S2)
    A, h = poly_design(Y, kind), len(Y) // 2
    W, *_ = np.linalg.lstsq(A[:h], S1[:h], rcond=None)
    resid = ((S2[h:] - A[h:] @ W) ** 2).mean(0)
    return 1.0 - (resid - noise) / np.clip(signal, 1e-30, None)
