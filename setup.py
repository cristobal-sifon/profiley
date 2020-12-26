from __future__ import (absolute_import, division, print_function)


import os
import re
from setuptools import find_packages, setup

# folder where pygmos is stored
here = os.path.abspath(os.path.dirname(__file__))

#this function copied from pip's setup.py
#https://github.com/pypa/pip/blob/1.5.6/setup.py
#so that the version is only set in the __init__.py and then read here
#to be consistent
def find_version(fname):
    version_file = read(fname)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


#Taken from the Python docs:
#Utility function to read the README file.
#Used for the long_description.  It's nice, because now 1) we have a
#top level README file and 2) it's easier to type in the README file
#than to put a raw string in below
def read(fname):
    fname = os.path.join(here, fname)
    if os.path.isfile(fname):
        return open(fname).read()


def read_requirements(reqfile):
    return [i for i in open(reqfile).read().split('\n') if i]


setup(
    name='profiley',
    version=find_version('src/profiley/__init__.py'),
    description='Calculation of common astrophysical profiles',
    author='Cristobal Sifon',
    author_email='cristobal.sifon@pucv.cl',
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    url='https://github.com/cristobal-sifon/profiley',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        ],
)
