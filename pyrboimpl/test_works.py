# Copyright 2015, 2016, 2017 Andrzej Cichocki

# This file is part of pyrbo.
#
# pyrbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyrbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyrbo.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import unittest, numpy as np, logging, time
from .leaf import turbo, T

log = logging.getLogger(__name__)

def pysum(n, x, y, out):
    for i in range(n):
        out[i] = x[i] + y[i]

def npsum(n, x, y, out):
    out[:n] = x[:n]
    out[:n] += y[:n]

@turbo(i = np.uint32, n = np.uint32, x = [np.float32], y = [np.float32], out = [np.float32])
def tsum(n, x, y, out):
    for i in range(n):
        out[i] = x[i] + y[i]

@turbo(types = dict(i = np.uint32, n = np.uint32, x = [T], y = [T], out = [T]), dynamic = True)
def gsum(n, x, y, out):
    for i in range(n):
        out[i] = x[i] + y[i]

class Cls:

    @turbo(types = dict(self = {}, i = np.uint32, n = np.uint32, x = [T], y = [T], out = [T]), dynamic = True)
    def oogsum(self, n, x, y, out):
        for i in range(n):
            out[i] = x[i] + y[i]

class TestTurbo(unittest.TestCase):

    def test_works(self):
        n = 100000
        x = np.arange(n, dtype = np.float32)
        y = np.arange(n, dtype = np.float32) * 2
        expected = np.empty(n, dtype = np.float32)
        pysum(n, x, y, expected)
        for task in npsum, tsum, gsum[T, np.float32], gsum, Cls().oogsum, Cls().oogsum[T, np.float32]:
            actual = np.empty(n, dtype = np.float32)
            task(n, x, y, actual)
            self.assertTrue(np.array_equal(expected, actual))

class TestSpeed(unittest.TestCase):

    ntomaxreltime = {100: 1.2, 1000: 1.2} # I guess in small array case we see some setup overhead.

    def test_fastenough(self):
        trials = 20
        outliers = 2
        meanof = lambda v: sum(v) / len(v)
        nandtasktotimes = {}
        for n in range(2, 6):
            n = 10 ** n
            log.info("n: %s", n)
            x = np.arange(n, dtype = np.float32)
            y = np.arange(n, dtype = np.float32) * 2
            for task in npsum, tsum, gsum[T, np.float32]:
                log.info("task: %s", task)
                out = np.empty(n, dtype = np.float32)
                times = []
                for _ in range(trials):
                    t = time.time()
                    task(n, x, y, out)
                    times.append((time.time() - t) * 1000)
                times.sort()
                log.info("trials: %s", ' '.join("%.6f" % q for q in times))
                nandtasktotimes[n, task] = times[outliers:-outliers]
        fails = []
        for (n, task), times in nandtasktotimes.items():
            if npsum != task:
                maxreltime = self.ntomaxreltime.get(n, 1)
                reltime = meanof(times) / meanof(nandtasktotimes[n, npsum])
                if reltime > maxreltime:
                    fails.append("(n = %r, task = %r, reltime = %r, maxreltime = %r)" % (n, task, reltime, maxreltime))
        self.assertEqual([], fails)

@turbo(n = np.uint32, acc = np.uint32)
def triple(n):
    acc = 0
    for UNROLL in range(n):
        acc += 3
    return acc

class TestUnroll(unittest.TestCase):

    def test_unroll(self):
        self.assertEqual(21, triple(7))
