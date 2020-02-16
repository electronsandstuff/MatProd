import matprod
import numpy as np
import timeit
import functools as f

# Make a test array
arr = np.random.rand(2,2,10000)*1.06

# Test the speed of the new method
testfun_new = lambda: matprod.lprod(arr)
print("Execution Time New: {:.0f} us".format(timeit.timeit(testfun_new, number=10000)/10000*1e6))

# Test the speed of the old method
testfun_old = lambda: f.reduce(np.dot, arr.T).T
print("Execution Time Old: {:.0f} ms".format(timeit.timeit(testfun_old, number=100)/100*1e3))

# Make sure they are the same
print()
print('Relative Difference of Elements:')
print((testfun_new() - testfun_old())/testfun_old())
print(testfun_old())
