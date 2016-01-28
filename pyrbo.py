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

import pyximport, sys, os, logging
from pyrboimpl import pyrboimpl
from pyrboimpl.common import BadArgException, NoSuchVariableException, NoSuchPlaceholderException, AlreadyBoundException

BadArgException = BadArgException
NoSuchVariableException = NoSuchVariableException
NoSuchPlaceholderException = NoSuchPlaceholderException
AlreadyBoundException = AlreadyBoundException

log = logging.getLogger(__name__)

def pyxinstall():
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
    return pyrboimpl.Turbo(nametotypespec)
