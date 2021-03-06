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
import sys

k = None

@turbo(types = dict(i = int, k = X), groupsets = {X: [range(5, 10)]})
def f(i):
    return i + k

@turbo(types = dict(i = int, k = X), groupsets = {X: [[int, float]]})
def g(i, k):
    return i + k

class TestGroups(TestCase):

    def test_works(self):
        self.assertEqual(13, f[X, 6](7))
        self.assertIn(f"{__name__}_turbo.f_5to9", sys.modules)
        self.assertEqual(13, f[X, 5](8))
        self.assertEqual(13, f[X, 7](6))
        self.assertEqual(13, f[X, 8](5))
        self.assertEqual(13, f[X, 9](4))
        self.assertEqual(13, f[X, 10](3))
        self.assertIn(f"{__name__}_turbo.f_10", sys.modules)

    def test_types(self):
        self.assertEqual(13, g[X, int](6, 7))
        self.assertIn(f"{__name__}_turbo.g_floatETint", sys.modules)
        self.assertEqual(13, g[X, float](6, 7))
