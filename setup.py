from setuptools import setup, find_packages


setup(
    name="radlab-plwordnet",
    version="0.0.1",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
