#!/usr/bin/env python

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

from __future__ import division # Required by tests script.
import numpy as np
import unittest
from turbo import turbo, T

def pytrue(x, y):
    return x / y

def pyfloor(x, y):
    return x // y

@turbo([np.int32, np.float32], x = T, y = T)
def turbotrue(x, y):
    return x / y

@turbo([np.int32, np.float32], x = T, y = T)
def turbofloor(x, y):
    return x // y

class TestDivision(unittest.TestCase):

    def test_floats(self):
        for truediv in pytrue, turbotrue[np.float32]:
            self.assertEqual(1.5, truediv(4.5, 3))
            self.assertEqual(-1.5, truediv(-4.5, 3))
        for floordiv in pyfloor, turbofloor[np.float32]:
            self.assertEqual(1, floordiv(4.5, 3))
            self.assertEqual(-2, floordiv(-4.5, 3))

    def test_ints(self):
        # Native function same as for floats:
        self.assertEqual(1.5, pytrue(3, 2))
        self.assertEqual(-1.5, pytrue(-3, 2))
        # turbo function does round-to-zero:
        self.assertEqual(1, turbotrue[np.int32](3, 2))
        self.assertEqual(-1, turbotrue[np.int32](-3, 2))
        # Native function same as for floats:
        self.assertEqual(1, pyfloor(3, 2))
        self.assertEqual(-2, pyfloor(-3, 2))
        # turbo function does round-to-zero:
        self.assertEqual(1, turbofloor[np.int32](3, 2))
        self.assertEqual(-1, turbofloor[np.int32](-3, 2))

if '__main__' == __name__:
    unittest.main()
