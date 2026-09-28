"""
The symmetry argument behind the offset encoding, checked numerically.

phi -> phi + pi maps the tone amplitudes f_j -> (-1)^j f_j and leaves every |a_m|^2 unchanged,
so the time-averaged comb spectrum must satisfy   S(f1, f2, f3, f4) = S(-f1, f2, -f3, f4).

  signed encoding  f = eps * s~      : s~ and (-s1, s2, -s3, s4) are the SAME point for the readout
  offset encoding  f = eps * (1+s~)  : they are different drives, and the spectra differ

    python tests/test_translation_symmetry.py
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, OPERATING_POINT

op, eps, R = OPERATING_POINT, 0.6, 96
s = np.array([0.6, -0.4, 0.5, 0.3])
s_flip = s * np.array([-1, 1, -1, 1])


def spectra(f_rows):
    """Mean spectrum and its standard error for each drive row, from R independent rings each (T = 100)."""
    f = torch.as_tensor(np.repeat(np.array(f_rows), R, axis=0), dtype=torch.float32)
    sol = LLESolver(N=op["N"], dt=op["dt"], Delta=op["Delta"], d2=op["d2"])
    sol.set_drive(op["F0"], f)
    a = sol.random_state(len(f), generator=torch.Generator().manual_seed(0))
    a, _ = sol.evolve(a, int(100 / op["dt"]))
    a, I = sol.evolve(a, int(100 / op["dt"]), accumulate=True, sample_every=5)
    I = I.numpy().reshape(len(f_rows), R, -1)
    return I.mean(1), I.std(1) / np.sqrt(R)


def rms_z(m1, e1, m2, e2, lines):
    return float(np.sqrt((((m1 - m2) / np.sqrt(e1 ** 2 + e2 ** 2))[lines] ** 2).mean()))


lines = [m % op["N"] for m in range(-8, 9)]                   # the 17 lines the policy reads
M, E = spectra([eps * s, eps * s_flip, eps * (1 + s), eps * (1 + s_flip)])
z_signed = rms_z(M[0], E[0], M[1], E[1], lines)
z_offset = rms_z(M[2], E[2], M[3], E[3], lines)
print(f"signed encoding: S(s) vs S(-s1, s2, -s3, s4): rms z-score = {z_signed:6.2f}   (1 = identical within noise)")
print(f"offset encoding: S(s) vs S(-s1, s2, -s3, s4): rms z-score = {z_offset:6.2f}   (>> 1 = distinguishable)")
assert z_signed < 2.0 and z_offset > 5.0
print("OK")
