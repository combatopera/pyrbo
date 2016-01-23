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

import inspect, re, importlib, pyximport, sys, os, logging
from unroll import unroll

log = logging.getLogger(__name__)

def pyxinstall():
    conf = {'inplace': True, 'build_in_temp': False}
    try:
        import turboconf
        conf.update(turboconf.turboconf)
    except ImportError:
        print >> sys.stderr, 'turboconf module not found in:', os.environ.get('PYTHONPATH')
    log.debug("pyximport config: %s", conf)
    pyximport.install(**conf) # Note -O3 is apparently the default.
pyxinstall()
del pyxinstall

class Placeholder:

    isplaceholder = True

    def __init__(self, name):
        self.name = name

    def __eq__(self, that):
        return self.name == that.name

    def __hash__(self):
        return hash(self.name)

    def __cmp__(self, that):
        return cmp(self.name, that.name)

    def __repr__(self):
        return self.name # Just import it.

    def resolvedspec(self, variant):
        return variant[self]

allplaceholders = set(Placeholder(chr(i)) for i in xrange(ord('T'), ord('Z') + 1))
globals().update([p.name, p] for p in allplaceholders)

class Type:

    isplaceholder = False

    def __init__(self, t):
        self.t = t

    def resolvedspec(self, variant):
        return self

    def nameorobj(self):
        return self.t.__name__

class Obj:

    def __init__(self, o):
        self.o = o

    def nameorobj(self):
        return self.o

class TypeSpec:

    def __init__(self, typespec):
        self.typespec = typespec if typespec in allplaceholders else Type(typespec)

    def iterplaceholders(self):
        if self.typespec.isplaceholder:
            yield self.typespec

    def resolvedspec(self, variant):
        return self.typespec.resolvedspec(variant)

    def typename(self, variant):
        return self.typespec.resolvedspec(variant).nameorobj()

class Array(TypeSpec):

    def ispotentialconst(self):
        return False

    def cparam(self, variant, name):
        return "np.ndarray[np.%s_t] py_%s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isfuncparam):
        if isfuncparam:
            yield "cdef np.%s_t* %s = &py_%s[0]" % (self.typename(variant), name, name)
        else:
            yield "cdef np.%s_t* %s" % (self.typename(variant), name)

class Scalar(TypeSpec):

    def ispotentialconst(self):
        return self.typespec in allplaceholders

    def cparam(self, variant, name):
        return "np.%s_t %s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isfuncparam):
        if not isfuncparam:
            yield "cdef np.%s_t %s" % (self.typename(variant), name)

class NoSuchVariableException(Exception): pass

class Variant:

    def __init__(self, placeholders, paramtoarg = {}):
        self.suffix = ''.join("_%s" % a.nameorobj() for _, a in sorted(paramtoarg.iteritems()))
        self.placeholders = placeholders
        self.paramtoarg = paramtoarg

    def spinoff(self, param, arg):
        if param not in self.placeholders:
            raise Exception(param)
        paramtoarg = self.paramtoarg.copy()
        paramtoarg[param] = arg
        return Variant(self.placeholders, paramtoarg)

    def __getitem__(self, param):
        return self.paramtoarg[param]

class BaseFunction:

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
            consts = dict([name, self.nametotypespec[name].resolvedspec(variant).o] for name in self.constnames)
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
                file = open(filepath)
                try:
                    existingtext = file.read()
                finally:
                    file.close()
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
        return getattr(sys.modules[fqmodulename], functionname)

class Partial:

    def __init__(self, basefunc, variant):
        self.basefunc = basefunc
        self.variant = variant

    def __getitem__(self, (param, arg)):
        arg = Type(arg) if type == type(arg) else Obj(arg)
        return Partial(self.basefunc, self.variant.spinoff(param, arg))

    def res(self):
        return self.basefunc.getvariant(self.variant)

    def __call__(self, *args, **kwargs):
        return self.res()(*args, **kwargs)

class turbo:

    def __init__(self, **nametotypespec):
        self.nametotypespec = {}
        placeholders = set()
        for name, typespec in nametotypespec.iteritems():
            if list == type(typespec):
                elementtypespec, = typespec
                typespec = Array(elementtypespec)
            else:
                typespec = Scalar(typespec)
            self.nametotypespec[name] = typespec
            placeholders.update(typespec.iterplaceholders())
        self.variant = Variant(placeholders)

    def __call__(self, pyfunc):
        return Partial(BaseFunction(self.nametotypespec, pyfunc), self.variant)
