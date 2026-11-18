"""Unit test: the non-Limber early-exit tolerance, the FKEM hand-off
and the noise of the low-rank (SVD) engine.

The non-Limber real-space C_l switch a lens bin (gg) or a lens-source
pair (gs) to Limber once they agree with Limber within a tolerance; the
switch multipole moves with the parameters and puts steps into
finite-difference derivatives. init_nonlimber_tolerance sets the
tolerances (likelihood keys nonlimber_tolerance_gg = 0.002 and
nonlimber_tolerance_gs = 0.01 by default; 0 = no early exit). The
shared factory (cosmolike_core/cocoa_testing_suites.py,
nonlimber_tolerance_suite) documents the checks. On example2:

    This project's 3x2pt data vector is in Fourier space:
    C_gg_tomo_ells and C_gs_tomo_ells never take the early exit,
    so test 3 checks that the tolerance changes nothing and test
    4 is skipped; the FKEM hand-off (5) and the SVD noise (6)
    apply to the Fourier vector.

The stencil noise is the whitened fourth difference of five evaluations
(cosmolike_notebook_utils.fisher.get_stencil_noise, step h = 0.025) of
the w0, wa and ROMAN_KL_B1_10 columns.

Measured on 2026-11-18 (OMP 4), then bounded:
  - test 3: the vector does not change (no early exit in Fourier
    space);
  - test 5: FKEM caps 150/200 give w0 4.64e-3/5.79e-3 and
    wa 4.13e-4/4.08e-4; margin 0.1 (one-sided: the w0 noise
    grows with the cap, so it comes from the added non-Limber
    multipoles, not from the hand-off); bound 6e-3 at cap 150;
  - test 6: SVD/FKEM ratio at most 1.03 (w0, cap 200); margin 0.1.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_nonlimber_tolerance.py -s
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The bounds below come from this project's measurement (the values
# are in the comment of the module docstring above).
TestNonLimberTolerance = u.nonlimber_tolerance_suite(
    u, name="TestNonLimberTolerance", example="example2",
    bias_parameter="ROMAN_KL_B1_10",
    expected_defaults=(0.002, 0.01),
    change_bound=None,
    noise_bound=0.006,
    handoff_margin=0.1,
    svd_rule='not_larger',
    svd_margin=0.1)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
