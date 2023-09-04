import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('src'))
import fastpotential as fs

a1 = np.zeros((3,3), dtype=np.cdouble)
a1[0,0] = 1.0
a1[2,2] = 4.0
a2 = np.zeros((3,3),dtype=np.cdouble)
a2[0,0] = 2.0
a2[2,2] = 5.0
res = np.zeros((3,3),dtype=np.cdouble)



# Addition happens in-place for the result array; this is to avoid multiple allocations when
# solving the Schrödinger equation.
def test_addition():
    fs.add_test(a1,a2,res, 3)
    assert np.abs(3.0-res[0,0]) < 1e-4
    assert np.abs(9.0-res[2,2]) < 1e-4
    a1[0,0] = 1.0j
    fs.add_test(a1,a2,res, 3)
    assert np.abs(1.0j + 2.0 - res[0,0]) < 1e-4
    a1[0,0] = 1.0
res[:,:] = 0.0
def test_const_multiplication():
    fs.multiply_by_constant_test(a1,2.0,res,3)
    assert np.abs(res[0,0] - 2.0) < 1e-4
    assert np.abs(res[2,2] - 8.0) < 1e-4
    fs.multiply_by_constant_test(a1,1.0j,res,3)
    assert np.abs(res[0,0] - 1.0j) < 1e-4
    assert np.abs(res[2,2] - 4.0j) < 1e-4
