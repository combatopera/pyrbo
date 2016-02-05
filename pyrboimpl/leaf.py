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

import pyximport, sys, os, logging, pyrboimpl

def pyxinstall():
    log = logging.getLogger(__name__)
    conf = {'inplace': True, 'build_in_temp': False}
    try:
        import turboconf
        conf.update(turboconf.turboconf)
    except ImportError:
        print >> sys.stderr, 'turboconf module not found in:', os.environ.get('PYTHONPATH')
    log.debug("pyximport config: %s", conf) # XXX: Can we use pyxbld files instead?
    pyximport.install(**conf) # Note -O3 is apparently the default.
pyxinstall()
del pyxinstall

globals().update([p.name, p] for p in (pyrboimpl.Placeholder(chr(i)) for i in xrange(ord('T'), ord('Z') + 1)))

def turbo(**nametotypespec):
    return pyrboimpl.Decorator(nametotypespec)

class ClassVariant:

    @classmethod
    def create(cvcls, cls):
        placeholders = set()
        for member in cls.__dict__.itervalues():
            if isinstance(member, pyrboimpl.Partial):
                placeholders.update(member.decorated.placeholders)
        return cvcls(cls.__name__, placeholders, {})

    def __init__(self, basename, placeholders, paramtoarg):
        self.basename = basename
        self.placeholders = placeholders
        self.paramtoarg = paramtoarg

    def spinoff(self, param, arg):
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = pyrboimpl.Type(arg) if isinstance(arg, type) else pyrboimpl.Obj(arg)
        return ClassVariant(self.basename, self.placeholders, paramtoarg)

class basegeneric(type):

    def __getitem__(cls, (param, arg)):
        members = {}
        members['variant'] = variant = cls.variant.spinoff(param, arg)
        for name, member in cls.__dict__.iteritems():
            if isinstance(member, pyrboimpl.Partial) and param in member.variant.unbound:
                member = member[param, arg]
            members[name] = member
        words = [variant.basename]
        for param in sorted(variant.placeholders):
            words.append(variant.paramtoarg[param].discriminator() if param in variant.paramtoarg else '?')
        return basegeneric('_'.join(words), cls.__bases__, members)

class generic(basegeneric):

    def __new__(self, name, bases, members):
        cls = type.__new__(self, name, bases, members)
        cls.variant = ClassVariant.create(cls)
        return cls
