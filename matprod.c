/*
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
*/

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Number of dimensions for output of function
npy_intp out_dims[] = {2, 2};

static PyObject*
lprod (PyObject *dummy, PyObject *args){
    PyObject *arg1=NULL, *in_array=NULL;

    // Fetch the argument
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    // Convert to a numpy array
    in_array = PyArray_FROMANY(arg1, NPY_DOUBLE, 3, 3, NPY_ARRAY_IN_ARRAY);

    // Get the dimensions
    npy_intp *dims = PyArray_DIMS( (PyArrayObject *)in_array);

    // Check that we are the right size
    if( (dims[0] != 2) || (dims[1] != 2) || (dims[2] < 2) ){
        // Do some error stuff
        PyErr_SetString(PyExc_ValueError, "Input array must have shape (2,2,n) with n>2");
        return NULL;
    }

    // Create the output array
    PyObject *out_array = PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);

    // Grab the input/output data pointer
    double *in_matrix = (double *)PyArray_DATA( (PyArrayObject *)in_array);
    double *out_matrix = (double *)PyArray_DATA( (PyArrayObject *)out_array);

    // Initialize the matrix
    npy_intp d = dims[2];
    out_matrix[0] = in_matrix[0];
    out_matrix[1] = in_matrix[d];
    out_matrix[2] = in_matrix[2*d];
    out_matrix[3] = in_matrix[3*d];

    // For each matrix
    double ai, bi, ci, di;
    double aj, bj, cj, dj;
    npy_intp i;
    for(i=1; i<d; i++){
      // Pull values from the arrays
      ai = in_matrix[i];
      bi = in_matrix[d+i];
      ci = in_matrix[2*d+i];
      di = in_matrix[3*d+i];
      aj = out_matrix[0];
      bj = out_matrix[1];
      cj = out_matrix[2];
      dj = out_matrix[3];

      // Compute the product w/ return mat and dump it into there
      out_matrix[0] = ai*aj + bi*cj;
      out_matrix[1] = ai*bj + bi*dj;
      out_matrix[2] = aj*ci + cj*di;
      out_matrix[3] = bj*ci + di*dj;
    }

    // Return the new matrix
    return out_array;
}

static PyObject*
cumlprod (PyObject *dummy, PyObject *args){
    PyObject *arg1=NULL, *in_array=NULL;

    // Fetch the argument
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    // Convert to a numpy array
    in_array = PyArray_FROMANY(arg1, NPY_DOUBLE, 3, 3, NPY_ARRAY_IN_ARRAY);

    // Get the dimensions
    npy_intp *dims = PyArray_DIMS( (PyArrayObject *)in_array);

    // Check that we are the right size
    if( (dims[0] != 2) || (dims[1] != 2) || (dims[2] < 2) ){
        // Do some error stuff
        PyErr_SetString(PyExc_ValueError, "Input array must have shape (2,2,n) with n>2");
        return NULL;
    }

    // Create the output array
    npy_intp cumprod_out_dims[3];
    cumprod_out_dims[0] = 2;
    cumprod_out_dims[1] = 2;
    cumprod_out_dims[2] = dims[2];
    PyObject *out_array = PyArray_SimpleNew(3, cumprod_out_dims, NPY_DOUBLE);

    // Grab the input/output data pointer
    double *in_matrix = (double *)PyArray_DATA( (PyArrayObject *)in_array);
    double *out_matrix = (double *)PyArray_DATA( (PyArrayObject *)out_array);

    // Initialize the matrix
    npy_intp d = dims[2];
    out_matrix[0] = in_matrix[0];
    out_matrix[d] = in_matrix[d];
    out_matrix[2*d] = in_matrix[2*d];
    out_matrix[3*d] = in_matrix[3*d];

    // For each matrix
    double ai, bi, ci, di;
    double aj, bj, cj, dj;
    npy_intp i;
    for(i=1; i<d; i++){
      // Pull values from the arrays
      ai = in_matrix[i];
      bi = in_matrix[d+i];
      ci = in_matrix[2*d+i];
      di = in_matrix[3*d+i];
      aj = out_matrix[i-1];
      bj = out_matrix[d+i-1];
      cj = out_matrix[2*d+i-1];
      dj = out_matrix[3*d+i-1];

      // Compute the product w/ return mat and dump it into there
      out_matrix[i] = ai*aj + bi*cj;
      out_matrix[d+i] = ai*bj + bi*dj;
      out_matrix[2*d+i] = aj*ci + cj*di;
      out_matrix[3*d+i] = bj*ci + di*dj;
    }

    // Return the new matrix
    return out_array;
}

// Define help text for the functions
const char lprod_help[] = "    Returns the left product of the matrices Ms where Ms is a (2,2,n) numpy\n    array to be interpreted as n (2,2) matrices.  Call the matrices M0, M1, M2,\n    ... Mn which are indexed as Ms[:,:,0], Ms[:,:,1], Ms[:,:,2], ... Ms[:,:,n].\n    The output will be equivalent to Mn @ ... @ M2 @ M1 @ M0.\n\n    Parameters\n    ----------\n    Ms : array_like\n      The array of matrices to be multiplied together.  Must have shape (2,2,n)\n      with n>2.\n\n    Returns\n    -------\n    Prod : ndarray\n      The matrix product Mn @ ... @ M2 @ M1 @ M0.  Note: this array will be\n      returned in floating point data format.  Shape will be (2,2).\n\n    Examples\n    --------\n    >>> Ms = np.random.rand(2,2,100)\n    >>> prod = matprod.lprod(Ms)";
const char cumlprod_help[] = "    Returns the cumulative left product of the matrices Ms where Ms is a (2,2,n)\n    numpy array to be interpreted as n (2,2) matrices.  Call the matrices M0,\n    M1, M2, ... Mn which are indexed as Ms[:,:,0], Ms[:,:,1], Ms[:,:,2], ...,\n    Ms[:,:,n].  The output will be the n matrices [M0, M1 @ M0, M2 @ M1 @ M0,\n    ..., Mn @ ... @ M2 @ M1 @ M0] contained in a (2,2,n) numpy array.\n\n    Parameters\n    ----------\n    Ms : array_like\n      The array of matrices to be multiplied together.  Must have shape (2,2,n)\n      with n>2.\n\n    Returns\n    -------\n    Prod : ndarray\n      The cumulative matrix product where Prod[:,:,0] = M0, Prod[:,:,1] =\n      M1 @ M0, Prod[:,:,0] = M2 @ M1 @ M0, etc.  Note: this will be\n      returned in floating point data format as a (2,2,n) shaped array.\n\n    Examples\n    --------\n    >>> Ms = np.random.rand(2,2,100)\n    >>> prod = matprod.cumlprod(Ms)";

// Our Module's Function Definition struct
static PyMethodDef myMethods[] = {
    { "lprod", lprod, METH_VARARGS, lprod_help},
    { "cumlprod", cumlprod, METH_VARARGS, cumlprod_help},
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "matprod",
    "Fast repeated multiplication of 2x2 matrices in a compiled numpy extension.",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_matprod(void){
    // Must be run or the extension will silently crash
    import_array();

    // Register the extension
    return PyModule_Create(&myModule);
}
