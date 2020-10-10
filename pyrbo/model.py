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

from .common import AlreadyBoundException, BadArgException, NoSuchPlaceholderException, NoSuchVariableException, NotDynamicException
from .unroll import unroll
from diapyr.util import innerclass, singleton
from functools import total_ordering
from importlib import import_module
from itertools import chain, product
from pathlib import Path
import inspect, logging, re, sys, threading

log = logging.getLogger(__name__)
threadstate = threading.local()

@singleton
class nocompile:

    def depth(self):
        return getattr(threadstate, 'compiledisabled', 0)

    def __enter__(self):
        threadstate.compiledisabled = self.depth() + 1

    def __exit__(self, *exc_info):
        threadstate.compiledisabled -= 1

class GroupSets:

    def __init__(self, groupsets):
        self.groupsets = groupsets

    def groups(self, param):
        return self.groupsets.get(param, ())

@total_ordering
class Placeholder:

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

class Arg:

    def spread(self, groups):
        for group in groups:
            if self.unwrap() in group:
                return [type(self)(u) for u in group]
        return [self]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.unwrap()!r})"

@total_ordering
class Type(Arg):

    isplaceholder = False

    def __init__(self, t):
        self.t = t

    def resolvedarg(self, variant):
        return self

    def typename(self):
        return self.t.__name__

    def discriminator(self):
        return self.typename()

    def groupdiscriminator(self, groups):
        return 'ET'.join(sorted(a.typename() for a in self.spread(groups)))

    def __lt__(self, that):
        return self.t < that.t # FIXME LATER: Does not work.

    def __eq__(self, that):
        return self.t == that.t

    def __hash__(self):
        return hash(self.t)

    def unwrap(self):
        return self.t

class Obj(Arg):

    def __init__(self, o):
        self.o = o

    def typename(self):
        raise BadArgException(self.o)

    def discriminator(self):
        return str(self.o)

    def groupdiscriminator(self, groups):
        for group in groups:
            if self.o in group:
                if range == type(group) and 1 == abs(group.step):
                    return f"{min(group)}to{max(group)}"
                return 'ET'.join(map(str, sorted(group)))
        return str(self.o)

    def unwrap(self):
        return self.o

class CDef:

    def __init__(self, name, text):
        self.name = name
        self.text = text

    def __str__(self):
        return self.text

class Array:

    def __init__(self, elementtypespec, ndim):
        self.ndimtext = f", ndim={ndim}" if 1 != ndim else ''
        self.zeros = ', '.join(['0'] * ndim)
        self.elementtypespec = elementtypespec

    def ispotentialconst(self):
        return False

    def cparam(self, variant, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        name = f"py_{name}"
        return CDef(name, f"np.ndarray[np.{elementtypename}_t{self.ndimtext}] {name}")

    def itercdefs(self, variant, name, isfuncparam):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        if isfuncparam:
            yield CDef(name, f"cdef np.{elementtypename}_t* {name} = &py_{name}[{self.zeros}]")
        else:
            yield CDef(name, f"cdef np.{elementtypename}_t* {name}")

    def iternestedcdefs(self, variant, undparent, dotparent, name):
        elementtypename = self.elementtypespec.resolvedarg(variant).typename()
        cname = f"{undparent}_{name}"
        pyname = f"py_{cname}"
        yield CDef(pyname, f"cdef np.ndarray[np.{elementtypename}_t{self.ndimtext}] {pyname} = {dotparent}.{name}")
        yield CDef(cname, f"cdef np.{elementtypename}_t* {cname} = &py_{undparent}_{name}[{self.zeros}]")

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
        return CDef(name, f"np.{typename}_t {name}")

    def itercdefs(self, variant, name, isfuncparam):
        if not isfuncparam:
            typename = self.typespec.resolvedarg(variant).typename()
            yield CDef(name, f"cdef np.{typename}_t {name}")

    def iternestedcdefs(self, variant, undparent, dotparent, name):
        typename = self.typespec.resolvedarg(variant).typename()
        cname = f"{undparent}_{name}"
        yield CDef(cname, f"cdef np.{typename}_t {cname} = {dotparent}.{name}")

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
        undparent = f"{undparent}_{name}"
        dotparent = f"{dotparent}.{name}"
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
            self.suffix = ''.join(f"_{arg.discriminator()}" for _, arg in sorted(paramtoarg.items()))
            self.groupsuffix = ''.join(f"_{arg.groupdiscriminator(decorated.groupsets.groups(param))}" for param, arg in sorted(paramtoarg.items()))
        self.paramtoarg = paramtoarg

    def spinoff(self, decorated, param, arg):
        if param not in decorated.placeholders:
            raise NoSuchPlaceholderException(param)
        if param in self.paramtoarg:
            raise AlreadyBoundException(param, self.paramtoarg[param].unwrap(), arg.unwrap())
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = arg
        return type(self)(decorated, paramtoarg)

    def complete(self, decorated, args):
        if not decorated.dynamic:
            raise NotDynamicException(decorated.name)
        paramtoarg = self.paramtoarg.copy()
        for param in self.unbound:
            paramtoarg[param] = decorated.placeholdertoresolver[param](args)
        return type(self)(decorated, paramtoarg)

    def groupvariants(self, decorated):
        def groupargs(param):
            return self.paramtoarg[param].spread(decorated.groupsets.groups(param))
        params = sorted(self.paramtoarg)
        for arglist in product(*(groupargs(param) for param in params)):
            yield type(self)(decorated, dict(zip(params, arglist)))

class Decorated:

    pyxbld = '''from distutils.extension import Extension
import numpy as np

def make_ext(name, source):
    return Extension(name, [source], include_dirs = [np.get_include()])
'''
    header = '''# cython: language_level=3

cimport numpy as np
import cython
'''
    template = """
@cython.boundscheck(False)
@cython.cdivision(True) # Don't check for divide-by-zero.
def %(name)s(%(cparams)s):
%(code)s"""
    deftemplate = "DEF %s = %r"
    eol = re.search(r'[\r\n]+', pyxbld).group()
    indentpattern = re.compile(r'^\s*')
    colonpattern = re.compile(r':\s*$')

    @classmethod
    def _getbody(cls, pyfunc):
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
        return bodyindent[functionindentlen:], ''.join(f"{line[functionindentlen:]}{cls.eol}" for line in lines[i:])

    def __init__(self, nametotypespec, dynamic, groupsets, pyfunc):
        co_varnames = pyfunc.__code__.co_varnames # The params followed by the locals.
        co_argcount = pyfunc.__code__.co_argcount
        self.paramnames = co_varnames[:co_argcount]
        self.localnames = [n for n in co_varnames[co_argcount:] if 'UNROLL' != n]
        self.constnames = []
        allnames = set(chain(self.paramnames, self.localnames))
        for name, typespec in nametotypespec.items():
            if name not in allnames:
                if not typespec.ispotentialconst():
                    raise NoSuchVariableException(name)
                self.constnames.append(name) # We'll make a DEF for it.
        self.fqmodule = pyfunc.__module__
        self.name = pyfunc.__name__
        self.bodyindent, self.body = self._getbody(pyfunc)
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
        self.groupsets = groupsets

    def getcomplete(self, variant):
        try:
            return self.suffixtocomplete[variant.suffix]
        except KeyError:
            self.suffixtocomplete[variant.suffix] = f = self.CompleteInfo(variant).load() # TODO: Do not cache Deferred.
            return f

    @innerclass
    class CompleteInfo:

        def __init__(self, variant):
            self.functionname = f"{self.name}{variant.suffix}"
            self.groupname = f"{self.name}{variant.groupsuffix}"
            self.fqmodulename = f"{self.fqmodule}_turbo.{self.groupname}"
            self.variant = variant

        def _updatefiles(self):
            def functiontext(variant):
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
                return self.template % dict(
                    name = f"{self.name}{variant.suffix}",
                    cparams = ', '.join(str(p) for p in cparams),
                    code = f"""{''.join(f"{self.bodyindent}{d}{self.eol}" for d in chain(defs, cdefs))}{body}""",
                )
            text = f"{self.header}{''.join(functiontext(v) for v in self.variant.groupvariants(self))}"
            fileparent = Path(sys.modules[self.fqmodule].__file__).parent / f"{self.fqmodule.split('.')[-1]}_turbo"
            fileparent.mkdir(exist_ok = True)
            (fileparent / '__init__.py').write_text('')
            (fileparent / f"{self.groupname}.pyx").write_text(text)
            (fileparent / f"{self.groupname}.pyxbld").write_text(self.pyxbld)

        def load(self):
            try:
                m = import_module(self.fqmodulename)
            except ImportError:
                self._updatefiles()
                compileenabled = not nocompile.depth()
                print('Compiling:' if compileenabled else 'Prepared:', self.groupname, file=sys.stderr)
                if not compileenabled:
                    return Deferred(self.fqmodulename, self.functionname)
                m = import_module(self.fqmodulename)
            return Complete(getattr(m, self.functionname))

    def __repr__(self):
        return f"{type(self).__name__}(<function {self.name}>)"

class BaseComplete:

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self.f(instance, *args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self.f!r})"

class Complete(BaseComplete):

    def __init__(self, f):
        self.f = f

class Deferred(BaseComplete):

    @property
    def f(self):
        return self._getf()

    def __init__(self, modulename, functionname):
        self.modulename = modulename
        self.functionname = functionname

    def _getf(self):
        assert self.modulename in sys.modules or not nocompile.depth()
        f = getattr(import_module(self.modulename), self.functionname)
        self._getf = lambda: f
        return f

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

class Partial:

    def __init__(self, decorated, variant):
        self.decorated = decorated
        self.variant = variant

    def todynamic(self):
        return Partial(self.decorated, self.variant, True)

    def __getitem__(self, paramandarg):
        param, arg = paramandarg
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        return partialorcomplete(self.decorated, self.variant.spinoff(self.decorated, param, arg))

    def __call__(self, *args, **kwargs):
        return self.decorated.getcomplete(self.variant.complete(self.decorated, args))(*args, **kwargs)

    def __get__(self, instance, owner):
        return InstancePartial(instance, self.decorated, self.variant)

    def __repr__(self):
        return f"{type(self).__name__}({self.decorated!r})"

class InstancePartial:

    def __init__(self, instance, decorated, variant):
        self.instance = instance
        self.decorated = decorated
        self.variant = variant

    def __call__(self, *args, **kwargs):
        return self.decorated.getcomplete(self.variant.complete(self.decorated, (self.instance,) + args))(self.instance, *args, **kwargs)

    def __getitem__(self, paramandarg):
        param, arg = paramandarg
        arg = Type(arg) if isinstance(arg, type) else Obj(arg)
        variant = self.variant.spinoff(self.decorated, param, arg)
        if variant.unbound:
            return InstancePartial(self.instance, self.decorated, variant)
        else:
            return InstanceComplete(self.instance, self.decorated.getcomplete(variant))

class Decorator:

    def __init__(self, nametotypespec, dynamic, groupsets):
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
        self.groupsets = GroupSets(groupsets)

    def __call__(self, pyfunc):
        decorated = Decorated(self.nametotypespec, self.dynamic, self.groupsets, pyfunc)
        return partialorcomplete(decorated, Variant(decorated, {}))
