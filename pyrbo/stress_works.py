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

from .leaf import T
from .test_works import gsum, npsum, tsum
from foyndation import invokeall
from functools import partial
from statistics import median
from unittest import TestCase
import numpy as np, sys, time

def _stderr(*args):
    print(*args, file = sys.stderr)

class TestSpeed(TestCase):

    class Task:

        coarse = 13
        fine = 15

        def __init__(self, task):
            self.task = task

        def _onetime(self, args):
            r = range(self.fine)
            mark = time.time()
            for _ in r:
                self.task(*args)
            return (time.time() - mark) / self.fine

        def gettime(self, *args):
            return median(self._onetime(args) for _ in range(self.coarse))

    sizes = [10 ** exp for exp in range(8) if exp not in {4, 5}]
    maxratio = 1
    reftask = Task(npsum)
    tasks = list(map(Task, [tsum, gsum[T:np.float32]]))

    def _ratios(self, size):
        x = np.arange(size, dtype = np.float32)
        y = np.arange(size, dtype = np.float32) * 2
        out = np.empty(size, dtype = np.float32)
        reftime = self.reftask.gettime(size, x, y, out)
        for task in self.tasks:
            r = task.gettime(size, x, y, out) / reftime
            _stderr(r)
            yield r

    def test_fastenough(self):
        invokeall(partial(self.assertLessEqual, ratio, self.maxratio) for size in self.sizes for ratio in self._ratios(size))
