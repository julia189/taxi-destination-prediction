
8
9
#This is what your ‘setup.py’ file should look like.
 
from setuptools import setup, find_packages
 
setup(
    setup_requires=['wheel'],
    name="dataingestion", #Name
    version="1.0", #Version
    packages = find_packages(),  # Automatically find the packages that are recognized in the '__init__.py'.
    install_requires=['pandas']
)