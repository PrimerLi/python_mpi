#!/usr/bin/env python-mpi

import numpy
from mpi4py import MPI
from scipy.sparse import coo_matrix
import cProfile

def sparseMatrix(dimension, l):
    row = []
    col = []
    data = []
    row.append(0)
    col.append(0)
    row.append(0)
    col.append(1)
    data.append(1+2*l)
    data.append(-l)
    for i in range(1, dimension - 1):
        for j in range(3):
            row.append(i)
            col.append(i + j - 1)
        data.append(-l)
        data.append(1+2*l)
        data.append(-l)
    row.append(dimension - 1)
    col.append(dimension - 2)
    row.append(dimension - 1)
    col.append(dimension - 1)
    data.append(-l)
    data.append(1+2*l)
    return dimension, row, col, data

def printSparseMatrix(dimension, l):
    dimension, row, col, data = sparseMatrix(dimension, l)
    A = coo_matrix((data, (row, col)), shape = (dimension, dimension))
    matrix = A.toarray()
    print matrix

def createSparseMatrix(dimension, l):
    dimension, row, col, data = sparseMatrix(dimension, l)
    A = coo_matrix((data, (row, col)), shape = (dimension, dimension))
    return A.toarray()

def getBoundaryIndices(row, startRowNumber, endRowNumber):
    imin = 0
    imax = 0
    for i in range(len(row)):
        if (row[i] == startRowNumber):
            imin = i
            break
    #print "imin = ", imin
    for i in range(imin, len(row)):
        if (row[i] == endRowNumber):
            imax = i
    #print "imax = ", imax
    return imin, imax

def productParallel(dimension, row, col, data, x, comm):
    import sys
    if (dimension != len(x)):
        print "Dimension incompatible. "
        sys.exit(-1)

    rank = comm.Get_rank()
    nproc = comm.Get_size()
    lengthPerProc = dimension/nproc

    startIndex = rank*lengthPerProc
    if (rank < nproc - 1):
        endIndex = (rank + 1)*lengthPerProc - 1
    else:
        endIndex = dimension - 1

    imin, imax = getBoundaryIndices(row, startIndex, endIndex)
    y = numpy.zeros(dimension)
    i = imin
    while(i <= imax):
        index = i
        s = 0
        count = 0
        while (index < len(row) and row[index] == row[i]):
            s = s + data[index]*x[col[i] + count]
            index = index + 1
            count = count + 1
        y[row[i]] = s
        i = index

    for i in range(1, nproc):
        if (rank == i):
            comm.send(y[startIndex:endIndex+1], dest = 0, tag = 200 + i)
    if (rank == 0):
        yList = []
        yList.append(y[startIndex:endIndex+1])
        for i in range(1, nproc):
            temp = comm.recv(source = i, tag = 200 + i)
            yList.append(temp)
        resultVector = numpy.concatenate(yList)
    else:
        resultVector = None

    resultVector = comm.bcast(resultVector, root = 0)
    return resultVector

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

    x = numpy.zeros(dimension)
    for i in range(dimension):
        x[i] = i + 1
    dimension, row, col, data = sparseMatrix(dimension, l)
    y = productParallel(dimension, row, col, data, x, comm)

    if (False and rank == 0):
        print "Matrix vector product using product() function: "
        print y

    if (False and rank == 0):    
        A = createSparseMatrix(dimension, l)
        print "test of program: "
        print numpy.dot(y - numpy.dot(A, x), y - numpy.dot(A, x))
        #print "Matrix vector product using numpy.dot : "
        #print numpy.dot(A, x)
    pr.disable()
    pr.dump_stats("profile")
    pr.print_stats()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

