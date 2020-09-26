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

from .common import AlreadyBoundException, BadArgException, NoSuchPlaceholderException, NoSuchVariableException
from .model import nocompile
from .leaf import generic, LOCAL, turbo, T, U, V, W, X, Y, Z

assert AlreadyBoundException
assert BadArgException
assert NoSuchPlaceholderException
assert NoSuchVariableException
assert generic
assert not LOCAL
assert nocompile
assert turbo
assert T
assert U
assert V
assert W
assert X
assert Y
assert Z
