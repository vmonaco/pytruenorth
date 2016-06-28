#! /usr/bin/env python
#
# Copyright (C) 2016 Vinnie Monaco <contact@vmonaco.com>

import os, sys
from setuptools import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(compile(open('pytruenorth/version.py').read(),
                  'pytruenorth/version.py', 'exec'))

try:
    import numpy as np
except ImportError:
    import os.path
    import sys

    # get RTD running.
    class np:
        @staticmethod
        def get_include():
            return os.path.join(sys.prefix, 'include')


install_requires = [
    'numpy',
    'ujson',
]

tests_require = [

]

docs_require = [

]


setup_options = dict(
    name='pytruenorth',
    version=__version__,
    author='Vinnie Monaco',
    author_email='contact@vmonaco.com',
    description='Python interface to TrueNorth',
    packages=['pytruenorth'],
    long_description=read('README.txt'),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    ext_modules=[

    ],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require
    },
    package_data={
        "pytruenorth": [
            "../README.md",
            "../README.txt",
            "../LICENSE",
            "../MANIFEST.in",
        ]
    },
)

if __name__ == '__main__':
    setup(**setup_options)
