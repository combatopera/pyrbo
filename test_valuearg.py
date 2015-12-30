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

import numpy as np
import unittest
from turbo import turbo, T, U

y = None

@turbo(x = T, y = U)
def turbotuple(x):
    return x, y

class TestValueArg(unittest.TestCase):

    def test_works(self):
        self.assertEqual((-5, 6), turbotuple(T = np.int32, U = 6)(-5))

if '__main__' == __name__:
    unittest.main()
