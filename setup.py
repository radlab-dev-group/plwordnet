from setuptools import setup, find_packages

with open(".version", "r") as fh:
    file_version = fh.read().strip()

setup(
    name="plwordnet",
    version=file_version,
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "plwordnet-cli=apps.cli.plwordnet_cli:main",
            "plwordnet-milvus=apps.cli.plwordnet_milvus_cli:main",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
