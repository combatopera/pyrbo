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

from .leaf import turbo, X
from unittest import TestCase
import numpy as np

n = None

@turbo(n = np.uint32, acc = np.uint32, i = np.uint32)
def triple(n):
    acc = 0
    for UNROLL, i in range(n):
        acc += 3
    return acc * 1000 + n

@turbo(n = X, acc = np.uint32)
def triple_const():
    acc = 0
    for UNROLL in range(n):
        acc += 3
    return acc * 1000 + n

class TestUnroll(TestCase):

    def test_unroll(self):
        self.assertEqual(21_007, triple(7))
        self.assertEqual(381_127, triple(0x80 - 1))
        self.assertEqual(384_128, triple(0x80))
        self.assertEqual(387_129, triple(0x80 + 1))

    def test_const(self):
        self.assertEqual(300_100, triple_const[X, 100]())
