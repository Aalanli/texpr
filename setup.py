from setuptools import setup

from mypyc.build import mypycify

setup(
    name='gdsl',
    packages=['gdsl'],
    ext_modules=mypycify(['gdsl/__init__.py', 'gdsl/lang.py', 'gdsl/utils.py', 'gdsl/ir.py']),
    # install_requires=[
    # ]
)
