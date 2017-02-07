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

class BadArgException(Exception):

    def __init__(self, obj):
        Exception.__init__(self, obj)

class NoSuchVariableException(Exception):

    def __init__(self, name):
        Exception.__init__(self, name)

class NoSuchPlaceholderException(Exception):

    def __init__(self, param):
        Exception.__init__(self, param)

class AlreadyBoundException(Exception):

    def __init__(self, param, current, given):
        Exception.__init__(self, param, current, given)

class NotDynamicException(Exception):

    def __init__(self, name):
        Exception.__init__(self, name)
