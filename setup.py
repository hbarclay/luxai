import os
from setuptools import setup, find_packages

#Version of the software from luxai import __version__

setup(
    name="luxaiimpala",
    #version=__version__,
    author="",
    author_email="",
    description=("Gather the most resources and survive the night!"),
    license='All rights reserved',
    long_description="",
    classifiers=[],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
    test_suite='tests',
)
