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

import unittest, numpy as np
from .leaf import turbo

@turbo(
    i = np.uint32,
    n = np.uint32,
    x = [np.float32],
    y = [np.float32],
    out = [np.float32],
)
def tsum(
    n,
    x,
    y,
    out,
):
    for i in range(n):
        out[i] = x[i] + y[i]

class TestTurbo(unittest.TestCase):

    def test_works(self):
        n = 100000
        x = np.arange(n, dtype = np.float32)
        y = np.arange(n, dtype = np.float32) * 2
        actual = np.empty(n, dtype = np.float32)
        for task in tsum,:
            task(n, x, y, actual)
            self.assertTrue(np.array_equal(x + y, actual))
