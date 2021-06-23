import timeit
import numpy as np
from multiprocessing.pool import Pool
from scipy import spatial


# cKDTree implementation
def ckdTree():
    tree = spatial.cKDTree(targets, leafsize=50)
    return [tree.query(point, k=4) for point in sources]


# Initialization to transfer kdtree
def setKdTree(tree):
    global kdtree
    kdtree = tree

# Worker must not be in another function for multiprocessing


def multiprocKd_worker(point):
    return kdtree.query(point, k=4)


# cKDTree process pool implementation
def multiprocCKd():
    tree = spatial.cKDTree(targets, leafsize=50)

    pool = Pool(initializer=setKdTree, initargs=(tree,))
    return pool.map(multiprocKd_worker, sources)


if __name__ == "__main__":
    # define the number of points for the two arrays
    n_targets = 3200000
    n_sources = 3200000

    # pick some random points
    targets = np.random.rand(n_targets, 3) * 100
    sources = np.random.rand(n_sources, 3) * 100

    # 
    print('cKDTree:        %s' % timeit.Timer(
        lambda: ckdTree()).repeat(1, 1)[0])
    print('multiprocCKd:   %s' % timeit.Timer(
        lambda: multiprocCKd()).repeat(1, 1)[0])
    
    