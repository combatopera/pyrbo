#!/usr/bin/env runpy

# Copyright 2014 Andrzej Cichocki

# This file is part of turbo.
#
# turbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# turbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with turbo.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np, time
from pkg.mymod import tsum, gsum

def pysum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]

def npsum(n, x, y, out):
    out[:n] = x[:n]
    out[:n] += y[:n]

def main():
  quarter = 3
  for n in 100, 1000, 10000, 100000:
    print n
    x = np.arange(n, dtype = np.float32)
    y = np.arange(n, dtype = np.float32) * 2
    for task in npsum, tsum, gsum[np.float32], pysum:
        print task.__name__
        out = np.empty(n, dtype = np.float32)
        times = []
        for _ in xrange(quarter * 4):
            t = time.time()
            task(n, x, y, out)
            times.append(time.time() - t)
        times.sort()
        quartiles = [(times[quarter * q - 1] + times[quarter * q]) / 2 for q in [1, 2, 3]]
        if npsum == task:
            scale = quartiles[1]
            check = out
        elif not np.array_equal(check, out):
            raise Exception
        print ' '.join("%10.6f" % (t / scale) for t in quartiles)
    print check[:5], check[-5:]

if '__main__' == __name__:
    main()
