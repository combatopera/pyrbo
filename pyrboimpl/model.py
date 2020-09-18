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

import inspect, re, importlib, sys, os, logging, itertools, functools
from .common import BadArgException, NoSuchPlaceholderException, AlreadyBoundException, NotDynamicException, NoSuchVariableException
from .unroll import unroll

log = logging.getLogger(__name__)

@functools.total_ordering
class Placeholder(object):

    isplaceholder = True

    def __init__(self, name):
        self.name = name

    def __lt__(self, that):
        return self.name < that.name

    def __eq__(self, that):
        return self.name == that.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name # Just import it.

    def resolvedarg(self, variant):
        return variant.paramtoarg[self]

@functools.total_ordering
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

    def __lt__(self, that):
        return self.t < that.t

    def __eq__(self, that):
        return self.t == that.t

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

class CDef:

    def __init__(self, name, text):
        self.name = name
        self.text = text

    def __str__(self):
        return self.text

class Array:

    def __init__(self, elementtypespec, ndim):
        self.ndimtext = ", ndim=%s" % ndim if 1 != ndim else ''
        self.zeros = ', '.join(['0'] * ndim)
        self.elementtypespec = elementtypespec

    def ispotentialconst(self):
        return False

    def cparam(self, variant, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        name = 'py_' + name
        return CDef(name, "np.ndarray[np.%s_t%s] %s" % (elementtypename, self.ndimtext, name))

    def itercdefs(self, variant, name, isfuncparam):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        if isfuncparam:
            yield CDef(name, "cdef np.%s_t* %s = &py_%s[%s]" % (elementtypename, name, name, self.zeros))
        else:
            yield CDef(name, "cdef np.%s_t* %s" % (elementtypename, name))

    def iternestedcdefs(self, variant, undparent, dotparent, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        cname = undparent + '_' + name
        pyname = 'py_' + cname
        yield CDef(pyname, "cdef np.ndarray[np.%s_t%s] %s = %s.%s" % (elementtypename, self.ndimtext, pyname, dotparent, name))
        yield CDef(cname, "cdef np.%s_t* %s = &py_%s_%s[%s]" % (elementtypename, cname, undparent, name, self.zeros))

    def iterplaceholders(self):
        if self.elementtypespec.isplaceholder:
            yield self.elementtypespec, lambda arg: Type(arg.dtype.type)

class Scalar:

    def __init__(self, typespec):
        self.typespec = typespec

    def ispotentialconst(self):
        return self.typespec.isplaceholder

    def cparam(self, variant, name):
        typename = self.typespec.resolvedarg(variant).typename()
        return CDef(name, "np.%s_t %s" % (typename, name))

    def itercdefs(self, variant, name, isfuncparam):
        if not isfuncparam:
            typename = self.typespec.resolvedarg(variant).typename()
            yield CDef(name, "cdef np.%s_t %s" % (typename, name))

    def iternestedcdefs(self, variant, undparent, dotparent, name):
        typename = self.typespec.resolvedarg(variant).typename()
        cname = undparent + '_' + name
        yield CDef(cname, "cdef np.%s_t %s = %s.%s" % (typename, cname, dotparent, name))

    def resolvedobj(self, variant):
        return self.typespec.resolvedarg(variant).o

    def iterplaceholders(self):
        if self.typespec.isplaceholder:
            yield self.typespec, lambda arg: Type(type(arg))

class Composite:

    def __init__(self, fields):
        self.fields = sorted(fields.items())

    def cparam(self, variant, name):
        return CDef(name, name)

    def itercdefs(self, variant, name, isfuncparam):
        for field, fieldtype in self.fields:
            for cdef in fieldtype.iternestedcdefs(variant, name, name, field):
                yield cdef

    def iterplaceholders(self):
        for field, fieldtype in self.fields:
            for placeholder, resolver in fieldtype.iterplaceholders():
                yield placeholder, FieldResolver(field, resolver)

    def iternestedcdefs(self, variant, undparent, dotparent, name):
        undparent += '_' + name
        dotparent += '.' + name
        for field, fieldtype in self.fields:
            for cdef in fieldtype.iternestedcdefs(variant, undparent, dotparent, field):
                yield cdef

class FieldResolver:

    def __init__(self, field, resolver):
        self.field = field
        self.resolver = resolver

    def __call__(self, arg):
        return self.resolver(getattr(arg, self.field))

class PositionalResolver:

    def __init__(self, i, resolver):
        self.i = i
        self.resolver = resolver

    def __call__(self, args):
        return self.resolver(args[self.i])

class Variant:

    def __init__(self, decorated, paramtoarg):
        self.unbound = set(p for p in decorated.placeholders if p not in paramtoarg)
        if not self.unbound:
            self.suffix = ''.join('_' + arg.discriminator() for _, arg in sorted(paramtoarg.items()))
        self.paramtoarg = paramtoarg

    def spinoff(self, decorated, param, arg):
        if param not in decorated.placeholders:
            raise NoSuchPlaceholderException(param)
        if param in self.paramtoarg:
            raise AlreadyBoundException(param, self.paramtoarg[param].unwrap(), arg.unwrap())
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = arg
        return Variant(decorated, paramtoarg)

    def complete(self, decorated, args):
        if not decorated.dynamic:
            raise NotDynamicException(decorated.name)
        paramtoarg = self.paramtoarg.copy()
        for param in self.unbound:
            paramtoarg[param] = decorated.placeholdertoresolver[param](args)
        return Variant(decorated, paramtoarg)

class Decorated(object):

    pyxbld = '''from distutils.extension import Extension
import numpy as np

def make_ext(name, source):
    return Extension(name, [source], include_dirs = [np.get_include()])
'''
    template = '''# cython: language_level=3

cimport numpy as np
import cython

%(defs)s
@cython.boundscheck(False)
@cython.cdivision(True) # Don't check for divide-by-zero.
def %(name)s(%(cparams)s):
%(code)s'''
    deftemplate = '''DEF %s = %r
'''
    eol = re.search(r'[\r\n]+', pyxbld).group()
    indentpattern = re.compile(r'^\s*')
    colonpattern = re.compile(r':\s*$')

    @classmethod
    def getbody(cls, pyfunc):
        lines = inspect.getsource(pyfunc).splitlines()
        getindent = lambda: cls.indentpattern.search(lines[i]).group()
        i = 0
        functionindentlen = len(getindent())
        while True:
            i += 1 # No point checking (first line of) decorator.
            if cls.colonpattern.search(lines[i]) is not None:
                break
        i += 1
        bodyindent = getindent()
        if re.search(r'=\s*LOCAL\s*$', lines[i]) is not None:
            i += 1
        return bodyindent[functionindentlen:], ''.join(line[functionindentlen:] + cls.eol for line in lines[i:])

    def __init__(self, nametotypespec, dynamic, pyfunc):
        co_varnames = pyfunc.__code__.co_varnames # The params followed by the locals.
        co_argcount = pyfunc.__code__.co_argcount
        self.paramnames = co_varnames[:co_argcount]
        self.localnames = [n for n in co_varnames[co_argcount:] if 'UNROLL' != n]
        self.constnames = []
        allnames = set(itertools.chain(self.paramnames, self.localnames))
        for name, typespec in nametotypespec.items():
            if name not in allnames:
                if not typespec.ispotentialconst():
                    raise NoSuchVariableException(name)
                self.constnames.append(name) # We'll make a DEF for it.
        self.fqmodule = pyfunc.__module__
        self.name = pyfunc.__name__
        self.bodyindent, self.body = self.getbody(pyfunc)
        # Note placeholders includes those in consts, placeholdertoresolver does not:
        self.placeholders = set()
        for typespec in nametotypespec.values():
            for param, _ in typespec.iterplaceholders():
                self.placeholders.add(param)
        self.placeholdertoresolver = {}
        for i, name in enumerate(self.paramnames):
            for placeholder, resolver in nametotypespec[name].iterplaceholders():
                if placeholder not in self.placeholdertoresolver:
                    self.placeholdertoresolver[placeholder] = PositionalResolver(i, resolver)
        self.suffixtocomplete = {}
        self.nametotypespec = nametotypespec
        self.dynamic = dynamic

    def getcomplete(self, variant):
        try:
            return self.suffixtocomplete[variant.suffix]
        except KeyError:
            self.suffixtocomplete[variant.suffix] = f = self.loadcomplete(variant)
            return f

    def loadcomplete(self, variant):
        functionname = self.name + variant.suffix
        fqmodulename = self.fqmodule + '_turbo.' + functionname
        if fqmodulename not in sys.modules:
            cparams = []
            cdefs = []
            for name in self.paramnames:
                typespec = self.nametotypespec[name]
                cparams.append(typespec.cparam(variant, name))
                cdefs.extend(typespec.itercdefs(variant, name, True))
            cdefnames = set(cdef.name for cdef in cparams)
            cdefnames.update(cdef.name for cdef in cdefs)
            for name in self.localnames:
                if name not in cdefnames:
                    typespec = self.nametotypespec[name]
                    cdefs.extend(typespec.itercdefs(variant, name, False))
            defs = []
            consts = dict([name, self.nametotypespec[name].resolvedobj(variant)] for name in self.constnames)
            for item in consts.items():
                defs.append(self.deftemplate % item)
            body = []
            unroll(self.body, body, consts, self.eol)
            body = ''.join(body)
            text = self.template % dict(
                defs = ''.join(defs),
                name = functionname,
                cparams = ', '.join(str(p) for p in cparams),
                code = ''.join("%s%s%s" % (self.bodyindent, cdef, self.eol) for cdef in cdefs) + body,
            )
            bldtext = self.pyxbld
            fileparent = os.path.join(os.path.dirname(sys.modules[self.fqmodule].__file__), self.fqmodule.split('.')[-1] + '_turbo')
            filepath = os.path.join(fileparent, functionname + '.pyx')
            bldpath = filepath + 'bld'
            existingtext = readornone(filepath)
            existingbld = readornone(bldpath)
            if text != existingtext or bldtext != existingbld:
                try:
                    os.mkdir(fileparent)
                except OSError:
                    pass
                open(os.path.join(fileparent, '__init__.py'), 'w').close()
                with open(filepath, 'w') as g:
                    g.write(text)
                    g.flush()
                with open(bldpath, 'w') as g:
                    g.write(bldtext)
                    g.flush()
                print("Compiling:", functionname, file=sys.stderr)
            importlib.import_module(fqmodulename)
        return Complete(getattr(sys.modules[fqmodulename], functionname))

    def __repr__(self):
        return "%s(<function %s>)" % (type(self).__name__, self.name)

def readornone(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()

class Complete(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self.f(instance, *args, **kwargs)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.f)

class InstanceComplete:

    def __init__(self, instance, f):
        self.instance = instance
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(self.instance, *args, **kwargs)

def partialorcomplete(decorated, variant):
    if variant.unbound:
        return Partial(decorated, variant)
    else:
        return decorated.getcomplete(variant)

class Partial(object):

    def __init__(self, decorated, variant):
        self.decorated = decorated
        self.variant = variant

    def todynamic(self):
        return Partial(self.decorated, self.variant, True)

    def __getitem__(self, xxx_todo_changeme):
        (param, arg) = xxx_todo_changeme
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        return partialorcomplete(self.decorated, self.variant.spinoff(self.decorated, param, arg))

    def __call__(self, *args, **kwargs):
        return self.decorated.getcomplete(self.variant.complete(self.decorated, args))(*args, **kwargs)

    def __get__(self, instance, owner):
        return InstancePartial(instance, self.decorated, self.variant)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.decorated)

class InstancePartial:

    def __init__(self, instance, decorated, variant):
        self.instance = instance
        self.decorated = decorated
        self.variant = variant

    def __call__(self, *args, **kwargs):
        return self.decorated.getcomplete(self.variant.complete(self.decorated, (self.instance,) + args))(self.instance, *args, **kwargs)

    def __getitem__(self, xxx_todo_changeme1):
        (param, arg) = xxx_todo_changeme1
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        variant = self.variant.spinoff(self.decorated, param, arg)
        if variant.unbound:
            return InstancePartial(self.instance, self.decorated, variant)
        else:
            return InstanceComplete(self.instance, self.decorated.getcomplete(variant))

class Decorator:

    def __init__(self, nametotypespec, dynamic):
        def wrap(spec):
            return spec if isinstance(spec, Placeholder) else Type(spec)
        def iternametotypespec(nametotypespec):
            for name, typespec in nametotypespec.items():
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
        self.dynamic = dynamic

    def __call__(self, pyfunc):
        decorated = Decorated(self.nametotypespec, self.dynamic, pyfunc)
        return partialorcomplete(decorated, Variant(decorated, {}))
