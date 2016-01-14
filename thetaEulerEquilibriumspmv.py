#!/usr/bin/env python-mpi

import numpy as np
import sys
import cProfile
import spmv
import spmvcg

def F(x):
    return 1

def u0(x):
    return 0

def thetaEuler(nx, ntmax, theta):
    '''theta == 0 means forward Euler method, theta == 1 means backward Euler method. 0 <= theta <= 1. '''
    dx = 1.0/nx
    dt = 0.5*dx**2
    x = []
    for i in range(nx + 1):
        x.append(i*dx)
    u = np.zeros(nx + 1)
    dimension = nx - 1
    dimension, rowA, colA, dataA = spmv.sparseMatrix(dimension, theta*dt/dx**2)
    dimension, rowB, colB, dataB = spmv.sparseMatrix(dimension, -(1-theta)*dt/dx**2)

    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = F(x[i])

    tol = 0.0001
    iterMax = 100
    step = 0
    while(True):
        V = []
        for element in u[1:-1]:
            V.append(element)
        u[1:-1] = spmvcg.conjugateGradient(dimension, rowA, colA, dataA, spmv.product(dimension, rowB, colB, dataB, u[1:-1]) + dt*f[1:-1], tol, iterMax)
        step = step + 1
        residual = 0
        for i in range(len(V)):
            residual = residual + (V[i] - u[1:-1][i])**2
        #print step
        #print residual
        if (step > ntmax):
            break
        if (np.sqrt(residual) < 0.0000001):
            break
    return x, u

def main():
    import sys
    if (len(sys.argv) != 3):
        print "nx = argv[1], theta = argv[2].  "
        return -1

    pr = cProfile.Profile()
    pr.enable()
    nx = int(sys.argv[1])
    theta = float(sys.argv[2])
    if (theta < 0 or theta > 1):
        print "Theta should be in range [0, 1]. "
        return -1
    x, u = thetaEuler(nx, 100000, theta)
    ofile = open("thetaEuler.txt", "w")
    for i in range(len(x)):
        ofile.write(str(x[i]) + "    " + str(u[i]) + "\n")
    ofile.close()

    def exact(x):
        return -0.5*x*(x - 1)

    ofile = open("exact.txt", "w")
    for i in range(len(x)):
        ofile.write(str(x[i]) + "    " + str(exact(x[i])) + "\n")
    ofile.close()
    pr.disable()
    pr.dump_stats("profile")
    pr.print_stats()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
