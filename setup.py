import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="convolutionalcodes",
    version="0.0.1",
    author="Pep Zorro",
    author_email="pepelzorro@example.com",
    description=("Convolutional codes for Amaranth HDL"),
    license="BSD-2-Clause",
    keywords="amaranth radio viterbi convolutional",
    packages=["convolutionalcodes"],
    package_data={"": ["util/__init__.py", "util/test.py"]},
    install_requires=["amaranth"],
    long_description=read("README.md"),
)
