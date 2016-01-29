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

import inspect, re, importlib, sys, os, logging, itertools
from unroll import unroll
from common import BadArgException, NoSuchVariableException, NoSuchPlaceholderException, AlreadyBoundException

log = logging.getLogger(__name__)

class Placeholder(object):

    isplaceholder = True

    def __init__(self, name):
        self.name = name

    def __cmp__(self, that):
        return cmp(self.name, that.name)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name # Just import it.

    def resolvedarg(self, variant):
        return variant.paramtoarg[self]

class Type:

    isplaceholder = False

    def __init__(self, t):
        self.t = t

    def resolvedarg(self, variant):
        return self

    def typename(self):
        return self.t.__name__

    def discriminator(self):
        return self.typename()

    def __cmp__(self, that):
        return cmp(self.t, that.t)

    def __hash__(self):
        return hash(self.t)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.t)

    def unwrap(self):
        return self.t

class Obj:

    def __init__(self, o):
        self.o = o

    def typename(self):
        raise BadArgException(self.o)

    def discriminator(self):
        return str(self.o)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.o)

class Array:

    def __init__(self, elementtypespec, ndim):
        self.ndimtext = ", ndim=%s" % ndim if 1 != ndim else ''
        self.zeros = ', '.join(['0'] * ndim)
        self.elementtypespec = elementtypespec

    def ispotentialconst(self):
        return False

    def cparam(self, variant, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        return "np.ndarray[np.%s_t%s] py_%s" % (elementtypename, self.ndimtext, name)

    def itercdefs(self, variant, name, isfuncparam):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        if isfuncparam:
            yield "cdef np.%s_t* %s = &py_%s[%s]" % (elementtypename, name, name, self.zeros)
        else:
            yield "cdef np.%s_t* %s" % (elementtypename, name)

    def iternestedcdefs(self, variant, parent, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        yield "cdef np.ndarray[np.%s_t%s] py_%s_%s = %s.%s" % (elementtypename, self.ndimtext, parent, name, parent, name)
        yield "cdef np.%s_t* %s_%s = &py_%s_%s[%s]" % (elementtypename, parent, name, parent, name, self.zeros)

    def iterinferred(self, accept, arg):
        if self.elementtypespec in accept:
            yield self.elementtypespec, Type(arg.dtype.type)

    def iterplaceholders(self):
        if self.elementtypespec.isplaceholder:
            yield self.elementtypespec

class Scalar:

    def __init__(self, typespec):
        self.typespec = typespec

    def ispotentialconst(self):
        return self.typespec.isplaceholder

    def cparam(self, variant, name):
        typename = self.typespec.resolvedarg(variant).typename()
        return "np.%s_t %s" % (typename, name)

    def itercdefs(self, variant, name, isfuncparam):
        if not isfuncparam:
            typename = self.typespec.resolvedarg(variant).typename()
            yield "cdef np.%s_t %s" % (typename, name)

    def iternestedcdefs(self, variant, parent, name):
        typename = self.typespec.resolvedarg(variant).typename()
        yield "cdef np.%s_t %s_%s = %s.%s" % (typename, parent, name, parent, name)

    def iterinferred(self, accept, arg):
        if self.typespec in accept:
            yield self.typespec, Type(type(arg))

    def resolvedobj(self, variant):
        return self.typespec.resolvedarg(variant).o

    def iterplaceholders(self):
        if self.typespec.isplaceholder:
            yield self.typespec

class Composite:

    def __init__(self, fields):
        self.fields = fields

    def cparam(self, variant, name):
        return name

    def itercdefs(self, variant, name, isfuncparam):
        for field, fieldtype in sorted(self.fields.iteritems()):
            for cdef in fieldtype.iternestedcdefs(variant, name, field):
                yield cdef

    def iterinferred(self, accept, arg):
        for field, fieldtype in sorted(self.fields.iteritems()):
            for inferred in fieldtype.iterinferred(accept, getattr(arg, field)):
                yield inferred

    def iterplaceholders(self):
        for fieldtype in self.fields.itervalues():
            for placeholder in fieldtype.iterplaceholders():
                yield placeholder

class Variant:

    def __init__(self, decorated, paramtoarg):
        if len(paramtoarg) == len(decorated.placeholders):
            self.suffix = ''.join('_' + arg.discriminator() for _, arg in sorted(paramtoarg.iteritems()))
        else:
            self.suffix = None
        self.paramtoarg = paramtoarg

    def spinoff(self, decorated, param, arg):
        if param not in decorated.placeholders:
            raise NoSuchPlaceholderException(param)
        if param in self.paramtoarg:
            raise AlreadyBoundException(param)
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = arg
        return Variant(decorated, paramtoarg)

    def complete(self, decorated, args):
        unbound = set(p for p in decorated.placeholders if p not in self.paramtoarg)
        paramtoarg = self.paramtoarg.copy()
        for name, arg in zip(decorated.varnames, args):
            for p, t in decorated.nametotypespec[name].iterinferred(unbound, arg):
                if p in paramtoarg and paramtoarg[p] != t:
                    raise AlreadyBoundException(p, paramtoarg[p].unwrap(), t.unwrap())
                paramtoarg[p] = t
        return Variant(decorated, paramtoarg)

class Decorated:

    template = '''cimport numpy as np
import cython

%(defs)s
@cython.boundscheck(False)
@cython.cdivision(True) # Don't check for divide-by-zero.
def %(name)s(%(cparams)s):
%(code)s'''
    deftemplate = '''DEF %s = %r
'''
    eol = re.search(r'[\r\n]+', template).group()
    indentpattern = re.compile(r'^\s*')

    @classmethod
    def getbody(cls, pyfunc):
        lines = inspect.getsource(pyfunc).splitlines()
        getindent = lambda: cls.indentpattern.search(lines[i]).group()
        i = 0
        functionindentlen = len(getindent())
        while True:
            bodyindent = getindent()
            if len(bodyindent) != functionindentlen:
                break
            i += 1
        return bodyindent[functionindentlen:], ''.join(line[functionindentlen:] + cls.eol for line in lines[i:])

    def __init__(self, nametotypespec, pyfunc):
        # The varnames are the locals including params:
        self.varnames = [n for n in pyfunc.func_code.co_varnames if 'UNROLL' != n]
        self.constnames = []
        varnames = set(self.varnames)
        for name, typespec in nametotypespec.iteritems():
            if name not in varnames:
                if not typespec.ispotentialconst():
                    raise NoSuchVariableException(name)
                self.constnames.append(name) # We'll make a DEF for it.
        self.fqmodule = pyfunc.__module__
        self.name = pyfunc.__name__
        self.bodyindent, self.body = self.getbody(pyfunc)
        self.argcount = pyfunc.func_code.co_argcount
        self.placeholders = set(itertools.chain(*(typespec.iterplaceholders() for typespec in nametotypespec.itervalues())))
        self.nametotypespec = nametotypespec
        self.variants = {}

    def getvariant(self, variant):
        try:
            return self.variants[variant.suffix]
        except KeyError:
            self.variants[variant.suffix] = f = self.getvariantimpl(variant)
            return f

    def getvariantimpl(self, variant):
        functionname = self.name + variant.suffix
        fqmodulename = self.fqmodule + '_turbo.' + functionname
        if fqmodulename not in sys.modules:
            cparams = []
            cdefs = []
            for i, name in enumerate(self.varnames):
                typespec = self.nametotypespec[name]
                isfuncparam = i < self.argcount
                if isfuncparam:
                    cparams.append(typespec.cparam(variant, name))
                cdefs.extend(typespec.itercdefs(variant, name, isfuncparam))
            defs = []
            consts = dict([name, self.nametotypespec[name].resolvedobj(variant)] for name in self.constnames)
            for item in consts.iteritems():
                defs.append(self.deftemplate % item)
            body = []
            unroll(self.body, body, consts, self.eol)
            body = ''.join(body)
            text = self.template % dict(
                defs = ''.join(defs),
                name = functionname,
                cparams = ', '.join(cparams),
                code = ''.join("%s%s%s" % (self.bodyindent, cdef, self.eol) for cdef in cdefs) + body,
            )
            fileparent = os.path.join(os.path.dirname(sys.modules[self.fqmodule].__file__), self.fqmodule.split('.')[-1] + '_turbo')
            filepath = os.path.join(fileparent, functionname + '.pyx')
            if os.path.exists(filepath):
                with open(filepath) as f:
                    existingtext = f.read()
            else:
                existingtext = None
            if text != existingtext:
                try:
                    os.mkdir(fileparent)
                except OSError:
                    pass
                open(os.path.join(fileparent, '__init__.py'), 'w').close()
                with open(filepath, 'w') as g:
                    g.write(text)
                    g.flush()
                log.debug("Compiling: %s", functionname)
            importlib.import_module(fqmodulename)
        return Descriptor(getattr(sys.modules[fqmodulename], functionname))

class Descriptor(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self.f(instance, *args, **kwargs)

def partialorcomplete(decorated, variant):
    if variant.suffix is not None:
        return decorated.getvariant(variant)
    else:
        return Partial(decorated, variant)

class Partial(Descriptor):

    def __init__(self, decorated, variant):
        self.decorated = decorated
        self.variant = variant

    def __getitem__(self, (param, arg)):
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        return partialorcomplete(self.decorated, self.variant.spinoff(self.decorated, param, arg))

    def __call__(self, *args, **kwargs):
        return self.decorated.getvariant(self.variant.complete(self.decorated, args))(*args, **kwargs)

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self(instance, *args, **kwargs)

class Decorator:

    def __init__(self, nametotypespec):
        def wrap(spec):
            return spec if isinstance(spec, Placeholder) else Type(spec)
        def iternametotypespec(nametotypespec):
            for name, typespec in nametotypespec.iteritems():
                if list == type(typespec):
                    elementtypespec, = typespec
                    ndim = 1
                    while list == type(elementtypespec):
                        elementtypespec, = elementtypespec
                        ndim += 1
                    typespec = Array(wrap(elementtypespec), ndim)
                elif dict == type(typespec):
                    typespec = Composite(dict(iternametotypespec(typespec)))
                else:
                    typespec = Scalar(wrap(typespec))
                yield name, typespec
        self.nametotypespec = dict(iternametotypespec(nametotypespec))

    def __call__(self, pyfunc):
        decorated = Decorated(self.nametotypespec, pyfunc)
        return partialorcomplete(decorated, Variant(decorated, {}))
