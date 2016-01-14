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

def product(dimension, row, col, data, x):
    import sys
    if (dimension != len(x)):
        print "Dimension incompatible. "
        sys.exit(-1)

    y = numpy.zeros(dimension)
    i = 0
    while(i < len(row)):
        index = i
        s = 0
        count = 0
        while(index < len(row) and row[index] == row[i]):
            s = s + data[index]*x[col[i] + count]
            index = index + 1
            count = count + 1
        y[row[i]] = s
        i = index

    return y

def main():
    import sys
    if (len(sys.argv) != 3):
        print "Matrix dimension = argv[1], l = argv[2]. "
        return -1

    dimension = int(sys.argv[1])
    l = float(sys.argv[2])

    x = numpy.zeros(dimension)
    for i in range(dimension):
        x[i] = 1.2*i + 1
    dimension, row, col, data = sparseMatrix(dimension, l)
    y = product(dimension, row, col, data, x)
    print "Matrix vector product using product() function: "
    print y
    A = createSparseMatrix(dimension, l)
    print "Matrix vector product using numpy.dot : "
    print numpy.dot(A, x)

if __name__ == "__main__":
    import sys
    sys.exit(main())
