#!/usr/bin/env python

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

import numpy as np
import unittest
from pyrbo import turbo, T, U, X, AlreadyBoundException

@turbo(x = [T], y = [U], n = np.uint32)
def addxtoy(x, y, n):
    while n:
        n -= 1
        y[n] += x[n]

@turbo(x = X, y = X)
def add(x, y):
    return x + y

@turbo(x = [T], y = [T])
def noinfer(x, y):
    pass

class TestInfer(unittest.TestCase):

    def test_works(self):
        x = np.arange(10, dtype = np.int32) + 1
        y = np.arange(10, dtype = np.float32)
        addxtoy(x, y, 5)
        self.assertEqual(range(1, 11), list(x))
        self.assertEqual([1, 3, 5, 7, 9, 5, 6, 7, 8, 9], list(y))

    def test_works2(self):
        self.assertEqual(11, add(5, 6))

    def test_failure(self):
        x = np.arange(10, dtype = np.int32)
        y = np.arange(10, dtype = np.float32)
        try:
            noinfer(x, y)
            self.fail("Expected already bound.")
        except AlreadyBoundException, e:
            self.assertEqual((T, np.int32, np.float32), e.args)

if '__main__' == __name__:
    unittest.main()
