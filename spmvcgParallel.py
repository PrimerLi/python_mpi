#!/usr/bin/env python-mpi

import numpy as np
from scipy.sparse import coo_matrix
import cProfile
import spmvParallel
from mpi4py import MPI

def conjugateGradient(dimension, row, col, data, b, tol, jmax, comm, printResidual = False):
    import sys
    if (len(b) != dimension):
        print "Dimension incompatible. "
        sys.exit(-1)
    x0 = np.zeros(dimension)
    r0 = b - spmvParallel.productParallel(dimension, row, col, data, x0, comm)
    p0 = r0
    j = 0
    while(True):
        rj = r0
        pj = p0
        xj = x0
        AP = spmvParallel.productParallel(dimension, row, col, data, pj, comm)

        alpha = np.dot(rj, rj)/np.dot(pj, AP)
        x0 = xj + alpha*pj
        r0 = rj - alpha*AP
        beta = np.dot(r0, r0)/np.dot(rj, rj)
        p0 = r0 + beta*pj
        j = j + 1
        if (printResidual):
            print j
            print np.dot(r0, r0)
        if (j == jmax):
            break
        if (np.dot(r0, r0) < tol**2):
            break
    return x0

def main():
    import sys
    if (len(sys.argv) != 3):
        print "Matrix dimension = argv[1], l = argv[2]. "
        return -1
    pr = cProfile.Profile()
    pr.enable()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    dimension = int(sys.argv[1])
    l = float(sys.argv[2])
    dimension, row, col, data = spmvParallel.sparseMatrix(dimension, l)
    b = np.zeros(dimension)
    for i in range(dimension):
        b[i] = i + 1

    tol = 0.0001
    iterMax = 100
    x = conjugateGradient(dimension, row, col, data, b, tol, iterMax, comm)
    if (False and rank == 0):
        A = spmvParallel.createSparseMatrix(dimension, l)
        print "Matrix A: "
        print A
        print "Vector b: "
        print b
        print "Solution of Ax = b: "
        print x
        print "Ax = "
        print spmvParallel.productParallel(dimension, row, col, data, x, comm)
    pr.disable()
    pr.dump_stats("profile")
    pr.print_stats()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
