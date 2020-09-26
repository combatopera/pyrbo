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

from .common import AlreadyBoundException, NoSuchPlaceholderException
from .model import Decorator, Obj, Partial, Placeholder, Type
import initnative

del initnative
globals().update([p.name, p] for p in (Placeholder(chr(i)) for i in range(ord('T'), ord('Z') + 1)))
LOCAL = None

def turbo(**kwargs):
    if 'types' not in kwargs:
        kwargs = dict(types = kwargs)
    nametotypespec = kwargs['types']
    dynamic = kwargs.get('dynamic', False)
    groupsets = kwargs.get('groupsets', {})
    return Decorator(nametotypespec, dynamic, groupsets)

class ClassVariant:

    @classmethod
    def create(cvcls, cls):
        placeholders = set()
        for member in cls.__dict__.values():
            if isinstance(member, Partial):
                placeholders.update(member.decorated.placeholders)
        return cvcls(cls.__name__, placeholders, {})

    def __init__(self, basename, placeholders, paramtoarg):
        self.basename = basename
        self.placeholders = placeholders
        self.paramtoarg = paramtoarg

    def spinoff(self, param, arg):
        if param not in self.placeholders:
            raise NoSuchPlaceholderException(param)
        if param in self.paramtoarg:
            raise AlreadyBoundException(param, self.paramtoarg[param].unwrap(), arg.unwrap())
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = arg
        return ClassVariant(self.basename, self.placeholders, paramtoarg)

class basegeneric(type):

    def __getitem__(cls, paramandarg):
        param, arg = paramandarg
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        variant = cls.turbo_variant.spinoff(param, arg)
        members = {}
        for name, member in cls.__dict__.items():
            if isinstance(member, Partial) and param in member.variant.unbound:
                member = member[param, arg.unwrap()]
            members[name] = member
        members['turbo_variant'] = variant
        words = [variant.basename]
        for param in sorted(variant.placeholders):
            words.append(variant.paramtoarg[param].discriminator() if param in variant.paramtoarg else '?')
        return basegeneric('_'.join(words), cls.__bases__, members)

class generic(basegeneric):

    def __new__(self, name, bases, members):
        cls = basegeneric.__new__(self, name, bases, members)
        cls.turbo_variant = ClassVariant.create(cls)
        return cls
