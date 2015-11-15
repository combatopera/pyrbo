import inspect, re, importlib, pyximport, sys, os

pyximport.install(inplace = True, build_in_temp = False)

template = '''cimport numpy as np
import cython

@cython.boundscheck(False)
def %(name)s(%(params)s):
%(code)s'''

eol = re.search(r'[\r\n]+', template).group()

indentpattern = re.compile(r'^\s*')

def getbody(f):
    lines = inspect.getsource(f).splitlines()
    getindent = lambda: indentpattern.search(lines[i]).group()
    i = 0
    functionindentlen = len(getindent())
    while True:
        bodyindent = getindent()
        if len(bodyindent) != functionindentlen:
            break
        i += 1
    return bodyindent[functionindentlen:], ''.join(line[functionindentlen:] + eol for line in lines[i:])

class Array:

    def __init__(self, type):
        self.typename = type.__name__

    def param(self, name):
        return "np.ndarray[np.%s_t] py_%s" % (self.typename, name)

    def itercdefs(self, name, isparam):
        if isparam:
            yield "cdef np.%s_t* %s = &py_%s[0]" % (self.typename, name, name)
        else:
            yield "cdef np.%s_t* %s" % (self.typename, name)

class Scalar:

    def __init__(self, type):
        self.typename = type.__name__

    def param(self, name):
        return "np.%s_t %s" % (self.typename, name)

    def itercdefs(self, name, isparam):
        if not isparam:
            yield "cdef np.%s_t %s" % (self.typename, name)

def getpackagedot(f):
    p = (f.__module__ + '.').split('.')
    del p[-2]
    return '.'.join(p)

class NoSuchVariableException(Exception): pass

class Turbo:

    def __init__(self, nametotype):
        self.nametotypeinfo = {}
        for name, thetype in nametotype.iteritems():
            if list == type(thetype):
                elementtype, = thetype
                typeinfo = Array(elementtype)
            else:
                typeinfo = Scalar(thetype)
            self.nametotypeinfo[name] = typeinfo

    def __call__(self, f):
        varnames = set(f.func_code.co_varnames)
        for name in self.nametotypeinfo:
            if name not in varnames:
                raise NoSuchVariableException(name)
        params = []
        cdefs = []
        for i, name in enumerate(f.func_code.co_varnames):
            t = self.nametotypeinfo[name]
            expr = name
            param = i < f.func_code.co_argcount
            if param:
                params.append(t.param(name))
            cdefs.extend(t.itercdefs(name, param))
        functionname = f.__name__
        bodyindent, body = getbody(f)
        text = template % dict(name = functionname,
            params = ', '.join(params),
            code = ''.join("%s%s%s" % (bodyindent, cdef, eol) for cdef in cdefs) + body)
        packagedot = getpackagedot(f)
        fqmodulename = packagedot + 'turbo_' + functionname
        path = fqmodulename.replace('.', os.sep) + '.pyx'
        if os.path.exists(path):
            f = open(path)
            try:
                oldtext = f.read()
            finally:
                f.close()
        else:
            oldtext = None
        if text != oldtext:
            g = open(path, 'w')
            try:
                g.write(text)
                g.flush()
            finally:
                g.close()
        importlib.import_module(fqmodulename)
        return getattr(sys.modules[fqmodulename], functionname)

def turbo(**nametotype):
    return Turbo(nametotype)
