#! /usr/bin/env python3
'''
MatProd - Fast repeated multiplication of 2x2 matrices
Copyright (C) 2020 Christopher M. Pierce (contact@chris-pierce.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
################################################################################
# IMPORTS
################################################################################
import numpy as np
import timeit
import os
import sys
import pathlib
import unittest

# Try to add the recently built package to our path
if(os.path.isdir('build')):
    build_dir = os.listdir('build')
    bin_paths = [pathlib.Path.cwd()/ 'build' / x for x in build_dir]
    for bin_path in bin_paths:
        sys.path.append(str(bin_path.resolve()))

# Now try to import matprod
import matprod

################################################################################
# UNIT TESTS
################################################################################
class TestMatProd(unittest.TestCase):
    def test_lprod(self):
        # Make a test array
        arr = np.random.rand(2,2,100)

        # Compute the expected result
        reference_mat = arr[:,:,0]
        for mat in arr[:,:,1:].T:
            reference_mat = mat.T @ reference_mat

	# Call the library function
        test_mat = matprod.lprod(arr)

	# Compare
        self.assertTrue(np.isclose(reference_mat, test_mat).all())

    def test_rprod(self):
        # Make a test array
        arr = np.random.rand(2,2,100)

        # Compute the expected result with numpy (so many transposes...)
        reference_mat = [arr[:,:,0].T]
        for mat in arr[:,:,1:].T:
            reference_mat.append((mat.T @ reference_mat[-1].T).T)
        reference_mat = np.array(reference_mat).T

	# Call the library function
        test_mat = matprod.cumlprod(arr)

	# Compare
        self.assertTrue(np.isclose(reference_mat, test_mat).all())

################################################################################
# SCRIPT BEGINS HERE
################################################################################
if __name__ == "__main__":
    unittest.main()
