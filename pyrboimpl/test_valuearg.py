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
from .leaf import turbo, T, X, Y
from .common import BadArgException

y = None

@turbo(x = T, y = Y)
def turbotuple(x):
    return x, y

@turbo(v = [X])
def arrayof(v):
    pass

class TestValueArg(unittest.TestCase):

    def test_works(self):
        self.assertEqual((-5, 6), turbotuple[T, np.int32][Y, 6](-5))

    def test_arrayof(self):
        try:
            arrayof[X, 100]
            self.fail("Expected bad arg.")
        except BadArgException as e:
            self.assertEqual((100,), e.args)
