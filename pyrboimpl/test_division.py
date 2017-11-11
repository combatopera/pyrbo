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

from __future__ import division # Required by tests script.
import numpy as np, unittest
from .leaf import turbo, T

def pytrue(x, y):
    return x / y

def pyfloor(x, y):
    return x // y

@turbo(x = T, y = T)
def turbotrue(x, y):
    return x / y

@turbo(x = T, y = T)
def turbofloor(x, y):
    return x // y

class TestDivision(unittest.TestCase):

    def test_floats(self):
        for truediv in pytrue, turbotrue[T, np.float32]:
            self.assertEqual(1.5, truediv(4.5, 3))
            self.assertEqual(-1.5, truediv(-4.5, 3))
        for floordiv in pyfloor, turbofloor[T, np.float32]:
            self.assertEqual(1, floordiv(4.5, 3))
            self.assertEqual(-2, floordiv(-4.5, 3))

    def test_ints(self):
        # Native function same as for floats:
        self.assertEqual(1.5, pytrue(3, 2))
        self.assertEqual(-1.5, pytrue(-3, 2))
        # turbo function does round-to-zero:
        self.assertEqual(1, turbotrue[T, np.int32](3, 2))
        self.assertEqual(-1, turbotrue[T, np.int32](-3, 2))
        # Native function same as for floats:
        self.assertEqual(1, pyfloor(3, 2))
        self.assertEqual(-2, pyfloor(-3, 2))
        # turbo function does round-to-zero:
        self.assertEqual(1, turbofloor[T, np.int32](3, 2))
        self.assertEqual(-1, turbofloor[T, np.int32](-3, 2))
