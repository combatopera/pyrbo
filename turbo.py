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

typeparamtoindex = {}
for i, name in enumerate(xrange(ord('T'), ord('Z') + 1)):
    name = chr(name)
    exec "class %s: pass" % name
    typeparamtoindex[eval(name)] = i
nametotypeparam = dict([t.__name__, t] for t in typeparamtoindex)

def nameorobj(a):
    try:
        return a.__name__
    except AttributeError:
        return a

class Variable:

    def __init__(self, type):
        self.type = type

    def typename(self, variant):
        try:
            a = variant.args[self.type]
        except KeyError:
            a = self.type
        return nameorobj(a)

class Array(Variable):

    def isparam(self):
        return False

    def param(self, variant, name):
        return "np.ndarray[np.%s_t] py_%s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isparam):
        if isparam:
            yield "cdef np.%s_t* %s = &py_%s[0]" % (self.typename(variant), name, name)
        else:
            yield "cdef np.%s_t* %s" % (self.typename(variant), name)

class Scalar(Variable):

    def isparam(self):
        return self.type in typeparamtoindex

    def param(self, variant, name):
        return "np.%s_t %s" % (self.typename(variant), name)

    def itercdefs(self, variant, name, isparam):
        if not isparam:
            yield "cdef np.%s_t %s" % (self.typename(variant), name)

class NoSuchVariableException(Exception): pass

class Variant:

    def __init__(self, args):
        self.suffix = ''.join("_%s" % nameorobj(a) for _, a in sorted(args.iteritems()))
        self.args = args

class BaseFunction:

    template = '''cimport numpy as np
import cython

%(defs)s
@cython.boundscheck(False)
@cython.cdivision(True) # Don't check for divide-by-zero.
def %(name)s(%(params)s):
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

    def __init__(self, nametotypeinfo, pyfunc):
        self.varnames = [n for n in pyfunc.func_code.co_varnames if 'UNROLL' != n]
        self.constnames = []
        varnames = set(self.varnames)
        for name, typeinfo in nametotypeinfo.iteritems():
            if name not in varnames:
                if not typeinfo.isparam():
                    raise NoSuchVariableException(name)
                self.constnames.append(name)
        self.fqmodule = pyfunc.__module__
        self.name = pyfunc.__name__
        self.bodyindent, self.body = self.getbody(pyfunc)
        self.argcount = pyfunc.func_code.co_argcount
        self.nametotypeinfo = nametotypeinfo

    def getvariant(self, variant):
        functionname = self.name + variant.suffix
        fqmodulename = self.fqmodule + '_turbo.' + functionname
        if fqmodulename not in sys.modules:
            params = []
            cdefs = []
            for i, name in enumerate(self.varnames):
                typeinfo = self.nametotypeinfo[name]
                isparam = i < self.argcount
                if isparam:
                    params.append(typeinfo.param(variant, name))
                cdefs.extend(typeinfo.itercdefs(variant, name, isparam))
            defs = []
            consts = dict([name, self.nametotypeinfo[name].typename(variant)] for name in self.constnames)
            for item in consts.iteritems():
                defs.append(self.deftemplate % item)
            body = []
            unroll(self.body, body, consts, self.eol)
            body = ''.join(body)
            text = self.template % dict(
                defs = ''.join(defs),
                name = functionname,
                params = ', '.join(params),
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
                    open(os.path.join(fileparent, '__init__.py'), 'w').close()
                except OSError:
                    pass
                with open(filepath, 'w') as g:
                    g.write(text)
                    g.flush()
            importlib.import_module(fqmodulename)
        return getattr(sys.modules[fqmodulename], functionname)

class Lookup:

    def __init__(self, basefunc, typeargs, typeparams):
        self.basefunc = basefunc
        self.typeargs = typeargs
        self.typeparams = typeparams

    def __call__(self, **typeargsupdate):
        typeargs = self.typeargs.copy()
        for k, v in typeargsupdate.iteritems():
            if k not in nametotypeparam or nametotypeparam[k] not in self.typeparams:
                raise Exception(k)
            typeargs[nametotypeparam[k]] = v
        if len(typeargs) < len(self.typeparams):
            return Lookup(self.basefunc, typeargs, self.typeparams)
        return self.basefunc.getvariant(Variant(typeargs))

class Turbo:

    def __init__(self, nametotype):
        self.nametotypeinfo = {}
        self.typeparams = set()
        for name, thetype in nametotype.iteritems():
            if list == type(thetype):
                elementtype, = thetype
                typeinfo = Array(elementtype)
            else:
                typeinfo = Scalar(thetype)
            self.nametotypeinfo[name] = typeinfo
            if typeinfo.type in typeparamtoindex:
                self.typeparams.add(typeinfo.type)

    def __call__(self, pyfunc):
        basefunc = BaseFunction(self.nametotypeinfo, pyfunc)
        if self.typeparams:
            return Lookup(basefunc, {}, self.typeparams)
        return basefunc.getvariant(Variant({}))

def turbo(**nametotype):
    return Turbo(nametotype)
