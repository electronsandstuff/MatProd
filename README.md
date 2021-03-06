# MatProd
Fast repeated multiplication of 2x2 matrices in a compiled numpy extension.

[![Travis](https://travis-ci.com/electronsandstuff/MatProd.svg?branch=master)](https://travis-ci.com/electronsandstuff/MatProd)


## Why does this exist
This extension is the solution to a performance problem with the repeated
multiplication of 2x2 numpy matrices in python.  For some applications in
computational science, it is necessary to take the product of between 1E4 and
1E8 matrices in a performant manner.  In numpy this can be done with a for
loop.  However, this introduces a tight loop in python which adds significant
overhead to the task.  In addition, if the matrices being multiplied are small,
there is additional overhead in the for loops used for the matrix product that
may be avoided.  The performance difference between the python implementation
and this c implementation of repeated matrix multiplication is order of
magnitude 1,000x.

## Installation
This package is available through on PyPi.  Simply run `pip install matprod`.

## Usage
All functions in the library accept a numpy array of the n 2x2 matrices to be
multiplied together.  The array `Ms` should have the shape (2,2,n) where
`Ms[:,:,0]` is the first matrix, `Ms[:,:,1]` is the second, and so on.

Two functions are presently exported from the extension.  These are `lprod(Ms)`
and `cumlprod(Ms)` which perform the repeated matrix left product and the
cumulative left product of the matrices respectively.  By left product I mean
that the result of `lprod(Ms)` is equivalent to the code `Ms[:,:,n] @ ... @
Ms[:,:,2] @ Ms[:,:,1] @ Ms[:,:,0]` in numpy.  The cumulative left product will
also return all of the intermediate products in a (2,2,n) numpy array.  The
first element will be `ret[:,:,0] = Ms[:,:,0]`, the second and third will be
`ret[:,:,0] = Ms[:,:,1] @ Ms[:,:,0]`, `ret[:,:,0] = Ms[:,:,2] @ Ms[:,:,1] @
Ms[:,:,0]`, and so on.

Test the code out with this simple example:
```python
import matprod

# Create a set of matrices to multiply
Ms = np.random.rand(2,2,10000)

# Take the left product
print(matprod.lprod(Ms))

# Try out the cumulative product
print(matprod.cumlprod(Ms)[:,:,-1])
```

## Performance
The time taken to multiply 10,000 2x2 matrices by a python implementation and
this library can be compared with the following scripts.
```python
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
```

On my 2016-era laptop, the output of this scripts was
```
Execution Time New: 37 us
Execution Time Old: 25 ms

Relative Difference of Elements:
[[-6.56716376e-16 -1.34748247e-15]
 [-9.04050404e-16 -1.27529490e-15]]
```

This is a nearly three orders of magnitude speed-up over the python
implementation!  The results are also identical to machine precision.

## Reporting Issues and Feature Requests
Please file an issue on the projects github page [here](https://github.com/electronsandstuff/MatProd).

## Building and Testing
If you are modifying or contributing to the library please build through the
python system using `./setup.py build`.  On Windows you will need to have
Microsoft Visual Studio installed and on linux, please have gcc and the python
3 header files.  Once built, test the library by running `./test.py`.  Please
run this in a virtual environment where matprod is not already installed so
gaurantee that you are testing the active project and not a previously
installed version of the library.
