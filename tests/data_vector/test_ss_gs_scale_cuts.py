"""Unit test: the UV-fraction scale cut of C_ss and C_gs (cosmo2D_scuts).

The scale cut of arXiv:2603.13031 splits each Fourier spectrum C_l into
the part sourced by modes below a cut k_max/D(z) (the IR part) and the
rest (the UV part), and keeps a band power while the UV fraction
|C_UV|/(|C_UV| + |C_IR|) stays below alpha = 0.05. k_max is quoted at
z = 0 in h/Mpc, and D is the linear growth factor with D(0) = 1. The
kept bands of a spectrum are a prefix in l; its last band center is
l_max(k_max). This file checks the cut of cosmic shear C_ss (Limber,
the only shear projection in cosmolike) and of galaxy-galaxy lensing
C_gs on both projections adopt_limber_gs selects (Limber, and the FKEM
non-Limber split).

The checks (the shared factory fourier_ss_gs_scale_cut_suite of
cosmolike_core/cocoa_testing_suites.py), on the frozen 3x2pt fiducial:
  1. window off, the scale-cut drivers against the data-vector path,
     bit for bit;
  2. C_IR + C_UV = C;
  3. for C_ss with the intrinsic alignment off, C_IR agrees with an
     independent numpy Limber integral;
  4. at k_max = 20 h/Mpc the UV fraction is below alpha in every band;
  5. l_max does not decrease with k_max, and the kept bands form nested
     prefixes in l;
and the next ordinary evaluation after the cut gives the data vector
of before bit for bit.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_ss_gs_scale_cuts.py
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

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here.
TestSsGsScaleCut = u.fourier_ss_gs_scale_cut_suite(
    u, name="TestSsGsScaleCut", example="example2")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
