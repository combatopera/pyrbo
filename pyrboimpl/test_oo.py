#!/usr/bin/env pyven

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
from leaf import turbo, X, dynamic, T, U, Z, generic
from common import NoSuchPlaceholderException, AlreadyBoundException

self_u = self_x = None

class My:

    def __init__(self, x):
        self.x = x

    @turbo(self = dict(x = np.int8), y = np.int8, z = np.int8)
    def plus(self, y):
        z = self_x + y
        return z

    @dynamic
    @turbo(self = dict(x = X), y = X, z = X)
    def plus2(self, y):
        z = self_x + y
        return z

@turbo(obj = dict(field = int))
def fieldlocal(obj):
    obj_field = 6

class TestOO(unittest.TestCase):

    def test_works(self):
        my = My(5)
        self.assertEqual(11, my.plus(6))
        self.assertEqual(11, my.plus2(6))

    def test_fieldlocal(self):
        class Obj: pass
        obj = Obj()
        obj.field = 5
        fieldlocal(obj)
        self.assertEqual(5, obj.field)

class Buf:

    __metaclass__ = generic

    def __init__(self, u):
        self.u = u

    @dynamic
    @turbo(self = dict(u = [T]), i = np.uint32, j = np.uint32, v = U)
    def fillpart(self, i, j, v):
        while i < j:
            self_u[i] = v
            i += 1

class TestBuf(unittest.TestCase):

    def test_works(self):
        t = np.uint16
        u = np.int32
        tbuf = Buf[T, t]
        ubuf = Buf[U, u]
        names = ['Buf', 'Buf_uint16_?', 'Buf_?_int32', 'Buf_uint16_int32', 'Buf_uint16_int32']
        for bufcls in Buf, tbuf, ubuf, tbuf[U, u], ubuf[T, t]:
            self.assertEqual(names.pop(0), bufcls.__name__)
            v = np.zeros(10, dtype = t)
            buf = bufcls(v)
            buf.fillpart(4, 6, 5)
            self.assertEqual([0, 0, 0, 0, 5, 5, 0, 0, 0, 0], list(v))
        try:
            Buf[Z, None]
            self.fail('Expected no such placeholder.')
        except NoSuchPlaceholderException, e:
            self.assertEqual((Z,), e.args)
        try:
            tbuf[T, TestBuf]
            self.fail('Expected already bound.')
        except AlreadyBoundException, e:
            self.assertEqual((T, t, TestBuf), e.args)

if '__main__' == __name__:
    unittest.main()
