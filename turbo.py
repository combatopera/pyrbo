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

import inspect, re, importlib, pyximport, sys, os, itertools, logging

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

header = '''cimport numpy as np
import cython
'''

template = '''
@cython.boundscheck(False)
@cython.cdivision(True) # Don't check for divide-by-zero.
def %(name)s(%(params)s):
%(code)s'''

eol = re.search(r'[\r\n]+', template).group()

indentpattern = re.compile(r'^\s*')

def getbody(pyfunc):
    lines = inspect.getsource(pyfunc).splitlines()
    getindent = lambda: indentpattern.search(lines[i]).group()
    i = 0
    functionindentlen = len(getindent())
    while True:
        bodyindent = getindent()
        if len(bodyindent) != functionindentlen:
            break
        i += 1
    return bodyindent[functionindentlen:], ''.join(line[functionindentlen:] + eol for line in lines[i:])

typeparamtoindex = {}
for i, name in enumerate(xrange(ord('T'), ord('Z') + 1)):
    name = chr(name)
    exec "class %s: pass" % name
    typeparamtoindex[eval(name)] = i

class Variable:

    def __init__(self, type):
        self.type = type

    def typename(self, variant):
        try:
            t = variant.types[typeparamtoindex[self.type]]
        except KeyError:
            t = self.type
        return t.__name__

class Array(Variable):

    def param(self, variant, name):
        return "np.ndarray[np.%s_t] py_%s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isparam):
        if isparam:
            yield "cdef np.%s_t* %s = &py_%s[0]" % (self.typename(variant), name, name)
        else:
            yield "cdef np.%s_t* %s" % (self.typename(variant), name)

class Scalar(Variable):

    def param(self, variant, name):
        return "np.%s_t %s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isparam):
        if not isparam:
            yield "cdef np.%s_t %s" % (self.typename(variant), name)

def getpackagedot(pyfunc):
    p = (pyfunc.__module__ + '.').split('.')
    del p[-2]
    return '.'.join(p)

class NoSuchVariableException(Exception): pass

class Variant:

    def __init__(self, types):
        self.suffix = ''.join("_%s" % t.__name__ for t in types)
        self.types = types

class BaseFunction:

    def __init__(self, turbonames, pyfunc):
        varnames = set(pyfunc.func_code.co_varnames)
        for name in turbonames:
            if name not in varnames:
                raise NoSuchVariableException(name)
        self.name = pyfunc.__name__
        self.bodyindent, self.body = getbody(pyfunc)

class Turbo:

    def __init__(self, gtypelists, nametotype):
        self.nametotypeinfo = {}
        for name, thetype in nametotype.iteritems():
            if list == type(thetype):
                elementtype, = thetype
                typeinfo = Array(elementtype)
            else:
                typeinfo = Scalar(thetype)
            self.nametotypeinfo[name] = typeinfo
        self.variants = [Variant(v) for v in itertools.product(*gtypelists)]

    def __call__(self, pyfunc):
        basefunc = BaseFunction(self.nametotypeinfo, pyfunc)
        root = {}
        rootkey = None # Abuse this as a unit type.
        for variant in self.variants:
            functionname = basefunc.name + variant.suffix
            params = []
            cdefs = []
            for i, name in enumerate(pyfunc.func_code.co_varnames):
                typeinfo = self.nametotypeinfo[name]
                isparam = i < pyfunc.func_code.co_argcount
                if isparam:
                    params.append(typeinfo.param(variant, name))
                cdefs.extend(typeinfo.itercdefs(variant, name, isparam))
            text = header + (template % dict(name = functionname,
                params = ', '.join(params),
                code = ''.join("%s%s%s" % (basefunc.bodyindent, cdef, eol) for cdef in cdefs) + basefunc.body))
            fqmodulename = getpackagedot(pyfunc) + 'turbo_' + functionname
            path = os.path.join(os.path.dirname(sys.modules[pyfunc.__module__].__file__), "turbo_%s.pyx" % functionname)
            if os.path.exists(path):
                file = open(path)
                try:
                    existingtext = file.read()
                finally:
                    file.close()
            else:
                existingtext = None
            if text != existingtext:
                g = open(path, 'w')
                try:
                    g.write(text)
                    g.flush()
                finally:
                    g.close()
            importlib.import_module(fqmodulename)
            parent = root
            keys = (rootkey,) + variant.types
            for k in keys[:-1]:
                if k not in parent:
                    parent[k] = {}
                parent = parent[k]
            parent[keys[-1]] = getattr(sys.modules[fqmodulename], functionname)
        return root[rootkey]

def turbo(*gtypelists, **nametotype):
    return Turbo(gtypelists, nametotype)
