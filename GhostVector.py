#!/usr/bin/env python-mpi

import numpy as np
from mpi4py import MPI

class GhostVector:
    def __init__(self):
        pass 

    def createGhostVector(self, comm, globalSize):
        rank = comm.Get_rank()
        nproc = comm.Get_size()

        self.globalSize = globalSize
        lengthPerProc = self.globalSize/nproc
        startIndex = rank*lengthPerProc
        if (rank < nproc - 1):
            endIndex = (rank + 1)*lengthPerProc - 1
        else:
            endIndex = self.globalSize - 1
        self.localSize = endIndex - startIndex + 1
        if (rank == 0 or rank == nproc - 1):
            self.ghostSize = self.localSize + 1
        else:
            self.ghostSize = self.localSize + 2
        self.ghostVector = np.zeros(self.ghostSize)
        return self.ghostVector

    def getGlobalSize(self):
        return self.globalSize

    def getLocalSize(self):
        return self.localSize

    def getGhostSize(self):
        return self.ghostSize

    def exchangeGhostValues(self, comm):
        nproc = comm.Get_size()
        rank = comm.Get_rank()

        if (nproc == 1):
            pass
        else:
            left = rank - 1
            right = rank + 1
            if (rank > 0):
                req = comm.isend(self.ghostVector[1], dest = left)
                self.ghostVector[0] = comm.recv(source = left)
                MPI.Request.Wait(req)
            if (rank < nproc - 1):
                req = comm.isend(self.ghostVector[-2], dest = right)
                self.ghostVector[-1] = comm.recv(source = right)
                MPI.Request.Wait(req)
        
    def dotProduct(self, comm):
        nproc = comm.Get_size()
        rank = comm.Get_rank()
        if (nproc == 1):
            return np.dot(self.ghostVector, self.ghostVector)
        else:
            if (rank == 0):
                localProduct = np.dot(self.ghostVector[:-1], self.ghostVector[:-1])
            elif (rank == nproc - 1):
                localProduct = np.dot(self.ghostVector[1:], self.ghostVector[1:])
            else:
                localProduct = np.dot(self.ghostVector[1:-1], self.ghostVector[1:-1])
            globalProduct = comm.allreduce(localProduct, MPI.SUM)
            return globalProduct

    def gather(self, comm):
        nproc = comm.Get_size()
        rank = comm.Get_rank()

        for i in range(1, nproc):
            if (rank == i):
                comm.send(self.ghostVector, dest = 0, tag = 100 + i)

        if (rank == 0):
            ghostList = []
            ghostList.append(self.ghostVector)
            for i in range(1, nproc):
                temp = comm.recv(source = i, tag = 100 + i)
                ghostList.append(temp)
            self.totalGhostVector = np.concatenate(ghostList)
        else:
            self.totalGhostVector = None

    def gatherAll(self, comm):
        self.gather(comm)
        self.totalGhostVector = comm.bcast(self.totalGhostVector, root = 0)

def main():
    ghostVector = GhostVector()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    u = ghostVector.createGhostVector(comm, 20)
    u[1] = -100
    ghostVector.gatherAll(comm)
    print "rank = ", rank, "  ", ghostVector.totalGhostVector
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
