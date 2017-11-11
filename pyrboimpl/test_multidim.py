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

import numpy as np, unittest
from .leaf import turbo, T

@turbo(types = dict(a = [[T]]), dynamic = True)
def multidim(a):
    return a[0], a[1], a[2], a[3]

class TestMultiDim(unittest.TestCase):

    def test_works(self):
        a = np.empty((2, 2), dtype = np.float32)
        a[0, 0] = 1
        a[0, 1] = 2
        a[1, 0] = 3
        a[1, 1] = 4
        self.assertEqual((1, 2, 3, 4), multidim(a))
