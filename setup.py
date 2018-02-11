######################################################################
# Copyright (C) 2017-2018 Darryl Reeves, Jaakko Luttinen
#
# This file is licensed under the MIT License.
######################################################################


import os
import versioneer


meta = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'junctiontree', '_meta.py')) as fp:
    exec(fp.read(), meta)

NAME         = 'junctiontree'
DESCRIPTION  = 'Junction tree and belief propagation algorithms'
AUTHOR       = meta['__author__']
URL          = 'https://github.com/jluttine/junction-tree'
VERSION      = versioneer.get_version()
COPYRIGHT    = meta['__copyright__']


if __name__ == "__main__":

    import os
    import sys

    python_version = int(sys.version.split('.')[0])
    if python_version < 3:
        raise RuntimeError(
            "JunctionTree requires Python 3. You are running Python {0}."
            .format(python_version)
        )

    # Utility function to read the README file.
    def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

    from setuptools import setup, find_packages

    # Setup for BayesPy
    setup(
        install_requires = [
            "numpy"
        ],
        packages         = find_packages(),
        name             = NAME,
        version          = VERSION,
        author           = AUTHOR,
        description      = DESCRIPTION,
        url              = URL,
        long_description = read('README.rst'),
        cmdclass         = versioneer.get_cmdclass(),
        keywords         = [
            'probabilistic programming',
            'Bayesian networks',
            'graphical models',
        ],
        classifiers = [
            'Programming Language :: Python :: 3 :: Only',
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: {0}'.format(meta['__license__']),
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    )
