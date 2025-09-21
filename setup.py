from setuptools import setup, find_packages


setup(
    name="plwordnet",
    version="0.0.1",
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
