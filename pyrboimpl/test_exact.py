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
from .leaf import turbo, T

@turbo(v = [T])
def f(v):
    pass

@turbo(x = T)
def f2(x):
    pass

class TestExact(unittest.TestCase):

    def test_exact(self):
        dtypes = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float32, np.float64)
        for t in dtypes:
            for u in dtypes:
                g = f[T, t]
                v = np.empty(10, dtype = u)
                if t == u:
                    g(v)
                else:
                    try:
                        g(v)
                        self.fail('Expected value error.')
                    except ValueError:
                        pass
                g2 = f2[T, t]
                x = u(10)
                g2(x) # All conversions allowed, including narrowing.
