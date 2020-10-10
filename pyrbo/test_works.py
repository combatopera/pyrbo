# Copyright 2015, 2016, 2017, 2020 Andrzej Cichocki

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

from .leaf import turbo, T
from .model import Deferred, nocompile
from unittest import TestCase
import logging, numpy as np, os, time

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

class TestTurbo(TestCase):

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

    def test_deferredcompile(self):
        def loadtdiff():
            from .test_data.tdiff import tdiff
            return tdiff
        with nocompile:
            tdiff0 = loadtdiff()
        tdiff = loadtdiff() # XXX: Really not possible to compile here?
        self.assertIs(tdiff0, tdiff)
        n = 5
        x = np.arange(n, dtype = np.float32)
        y = np.arange(n, dtype = np.float32) * 2
        out = np.empty(n, dtype = np.float32)
        for _ in range(2):
            tdiff(n, x, y, out) # First time load module, second time use cached wrapper and function.
            self.assertEqual([0, -1, -2, -3, -4], list(out))

class TestDeferred(TestCase):

    def test_cache(self):
        d = Deferred(__name__, type(self).__name__)
        for _ in range(2):
            self.assertEqual(type(self), d.f)

    def test_nocompile(self):
        d = Deferred("%s.test_data.dnocompile" % __package__, 'dummy')
        with nocompile, self.assertRaises(AssertionError):
            d.f
        from .test_data import dnocompile
        del dnocompile
        with nocompile:
            d.f

class TestSpeed(TestCase):

    ntomaxreltime = {10000: 1.5, 100000: 1.5} if 'true' == os.environ.get('TRAVIS') else {}

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

class TestUnroll(TestCase):

    def test_unroll(self):
        self.assertEqual(21, triple(7))