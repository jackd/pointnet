from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages


with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')


setup(
    name='pointnet',
    version='0.0.1',
    description=('Reimplementation of pointnet in tensorflow'),
    url='http://github.com/jackd/weighpoint',
    author='Dominic Jack',
    author_email='thedomjack@gmail.com',
    license='MIT',
    packages=find_packages(),
    requirements=install_requires,
    zip_safe=True
)
