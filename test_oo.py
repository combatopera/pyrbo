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

import unittest, numpy as np
from pyrbo import turbo

self_x = None

class My:

    def __init__(self, x):
        self.x = x

    @turbo(self = dict(x = np.int8), y = np.int8, z = np.int8)
    def plus(self, y):
        z = self_x + y
        return z

class TestOO(unittest.TestCase):

    def test_works(self):
        my = My(5)
        self.assertEqual(11, my.plus(6))

if '__main__' == __name__:
    unittest.main()
