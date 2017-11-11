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
from .leaf import turbo, T, U, X

@turbo(types = dict(x = [T], y = [U], n = np.uint32), dynamic = True)
def addxtoy(x, y, n):
    while n:
        n -= 1
        y[n] += x[n]

@turbo(types = dict(x = X, y = X), dynamic = True)
def add(x, y):
    return x + y

@turbo(types = dict(x = [T], y = [T]), dynamic = True)
def noinfer(x, y):
    pass

class TestInfer(unittest.TestCase):

    def test_works(self):
        x = np.arange(10, dtype = np.int32) + 1
        y = np.arange(10, dtype = np.float32)
        addxtoy(x, y, 5)
        self.assertEqual(list(range(1, 11)), list(x))
        self.assertEqual([1, 3, 5, 7, 9, 5, 6, 7, 8, 9], list(y))

    def test_works2(self):
        self.assertEqual(11, add(5, 6))

    def test_failure(self):
        x = np.arange(10, dtype = np.int32)
        y = np.arange(10, dtype = np.float32) # Type not used.
        try:
            noinfer(x, y)
            self.fail("Expected value error.")
        except ValueError as e:
            self.assertEqual(("Buffer dtype mismatch, expected 'int32_t' but got 'float'",), e.args)
