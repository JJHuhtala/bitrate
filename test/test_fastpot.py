import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../src'))
import fastpotential as fs

a1 = np.zeros((3,3), dtype="complex")
a1[0,0] = 1.0
a1[2,2] = 4.0
a2 = np.zeros((3,3),dtype="complex")
a2[0,0] = 2.0
a2[2,2] = 5.0
res = np.zeros((3,3),dtype="complex")



# Addition happens in-place for the result array; this is to avoid multiple allocations when
# solving the Schrödinger equation.
def test_addition():
    fs.add_test(a1,a2,res, 3)
    assert np.abs(3.0-res[0,0]) < 1e-4
    assert np.abs(9.0-res[2,2]) < 1e-4

res[:,:] = 0.0
def test_multiplication():
    fs.multiply_by_constant_test(a1,res,2.0,3)
    assert(res[0,0] - 2.0) < 1e-4
    assert(res[2,2] - 8.0) < 1e-4
    fs.multiply_by_constant_test(a1,res,1.0j,3)
    assert(res[0,0] - 2.0j) < 1e-4
    assert(res[2,2] - 8.0j) < 1e-4
