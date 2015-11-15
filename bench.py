#!/usr/bin/env python

import numpy as np, time
from pkg.mymod import sum as sum2

def sum1(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]

def main():
    n = 100000
    x = np.arange(n, dtype = np.float32)
    y = np.arange(n, dtype = np.float32) * 2
    for s in sum1, sum2:
        out = np.empty(n, dtype = np.float32)
        t = time.time()
        s(n, x, y, out)
        print time.time() - t
        print out[:5], out[-5:]

if '__main__' == __name__:
    main()
