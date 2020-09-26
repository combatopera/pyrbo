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

from io import StringIO
import re

pattern = re.compile(r'^(\s*)for\s+UNROLL\s+in\s+range\s*\(\s*([^\s]+)\s*\)\s*:\s*$')
indentregex = re.compile(r'^\s*')
maxchunk = 0x80

def unroll(body, g, consts, eol):
    f = StringIO(body)
    buffer = []
    while True:
        line = buffer.pop(0) if buffer else f.readline()
        if not line:
            break
        m = pattern.search(line)
        if m is None:
            g.append(line)
            continue
        outerindent = m.group(1)
        variable = m.group(2)
        line = f.readline()
        m = indentregex.search(line)
        innerindent = m.group()
        body = []
        while line.startswith(innerindent):
            body.append(line)
            line = f.readline()
        buffer.append(line)
        if variable in consts:
            for _ in range(consts[variable]):
                for line in body:
                    g.append(f"{outerindent}{line[len(innerindent):]}")
        else:
            mask = 0x01
            while mask < maxchunk:
                g.append("%sif %s & 0x%x:%s" % (outerindent, variable, mask, eol))
                for _ in range(mask):
                    for line in body:
                        g.append(line)
                mask <<= 1
            g.append("%swhile %s >= 0x%x:%s" % (outerindent, variable, maxchunk, eol))
            for _ in range(maxchunk):
                for line in body:
                    g.append(line)
            g.append("%s%s -= 0x%x%s" % (innerindent, variable, maxchunk, eol))
