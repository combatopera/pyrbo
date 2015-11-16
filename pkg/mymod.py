import numpy as np
from turbo import turbo, T

@turbo(i = np.uint32, n = np.uint32, x = [np.float32], y = [np.float32], out = [np.float32])
def sum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]

@turbo([np.float32, np.int64], i = np.uint32, n = np.uint32, x = [T], y = [T], out = [T])
def gsum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]
