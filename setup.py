import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="convolutionalcodes",
    version="0.0.1",
    author="Pep Zorro",
    author_email="pepelzorro@example.com",
    description=("Convolutional codes for nMigen"),
    license="BSD-2-Clause",
    keywords="nmigen radio viterbi convolutional",
    packages=["convolutionalcodes"],
    long_description=read("README.md"),
)
