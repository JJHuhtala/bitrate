import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('src'))


# Addition happens in-place for the result array; this is to avoid multiple allocations when
# solving the Schrödinger equation.