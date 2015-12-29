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
from turbo import turbo, T

@turbo(i = np.uint32, n = np.uint32, x = [np.float32], y = [np.float32], out = [np.float32])
def tsum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]

@turbo(i = np.uint32, n = np.uint32, x = [T], y = [T], out = [T])
def gsum(n, x, y, out):
    for i in xrange(n):
        out[i] = x[i] + y[i]
