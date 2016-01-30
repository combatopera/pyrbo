#!/usr/bin/env pyven

# Copyright 2014 Andrzej Cichocki

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
import numpy as np, time, unittest, logging
from pkg.mymod import tsum, gsum, T

log = logging.getLogger(__name__)

def npsum(n, x, y, out):
    out[:n] = x[:n]
    out[:n] += y[:n]

class TestSpeed(unittest.TestCase):

    ntomaxreldiff = {100: .1} # I guess in small array case we see some setup overhead.

    def test_fastenough(self):
        quarter = 3
        nandtasktoquartiles = {}
        for n in 100, 1000, 10000, 100000:
            log.info("n: %s", n)
            x = np.arange(n, dtype = np.float32)
            y = np.arange(n, dtype = np.float32) * 2
            for task in npsum, tsum, gsum[T, np.float32]:
                log.info("task: %s", task)
                out = np.empty(n, dtype = np.float32)
                times = []
                for _ in xrange(quarter * 4):
                    t = time.time()
                    task(n, x, y, out)
                    times.append(time.time() - t)
                times.sort()
                quartiles = [(times[quarter * q - 1] + times[quarter * q]) / 2 for q in [1, 2, 3]]
                log.info("quartiles: %s", ' '.join("%.9f" % q for q in quartiles))
                nandtasktoquartiles[n, task] = quartiles
        for (n, task), quartiles in nandtasktoquartiles.iteritems():
            if npsum != task:
                maxreldiff = self.ntomaxreldiff.get(n, 0)
                for qi, tq in enumerate(quartiles):
                    npq = nandtasktoquartiles[n, npsum][qi]
                    reldiff = (tq - npq) / tq
                    self.assertTrue(reldiff <= maxreldiff, "(n = %r, task = %r, quartile = %r, reldiff = %r, maxreldiff = %r)" % (n, task, 1+qi, reldiff, maxreldiff))

if '__main__' == __name__:
    unittest.main()
