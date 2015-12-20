#!/bin/bash

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

set -ex

condaversion=3.16.0

cd ..

for project in runpy; do

    hg clone https://bitbucket.org/combatopera/$project

done

PATH="$PWD/runpy:$PATH"

wget --no-verbose http://repo.continuum.io/miniconda/Miniconda-$condaversion-Linux-x86_64.sh

bash Miniconda-$condaversion-Linux-x86_64.sh <<<$'\nyes\nminiconda\nno\n'

miniconda/bin/conda install -q pyflakes nose numpy cython

export MINICONDA_HOME="$PWD/miniconda"

cd -

export PYTHONPATH="$PWD/drone"

tests
