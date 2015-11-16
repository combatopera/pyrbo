#!/usr/bin/env python

import numpy as np, time
from pkg.mymod import tsum, gsum

def pysum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]

def npsum(n, x, y, out):
    out[:n] = x[:n]
    out[:n] += y[:n]

def main():
    n = 500000
    x = np.arange(n, dtype = np.float32)
    y = np.arange(n, dtype = np.float32) * 2
    for task in pysum, npsum, tsum, gsum[np.float32]:
        print task.__name__
        out = np.empty(n, dtype = np.float32)
        times = []
        for _ in xrange(8):
            t = time.time()
            task(n, x, y, out)
            times.append(time.time() - t)
        times.sort()
        q1 = (times[1] + times[2]) / 2
        q2 = (times[3] + times[4]) / 2
        q3 = (times[5] + times[6]) / 2
        print ' '.join("%.6f" % t for t in [q1, q2, q3])
        print out[:5], out[-5:]

if '__main__' == __name__:
    main()
