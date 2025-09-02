__SHORT__DESCRIPTION__ = """
PLWordNet Handler - narzędzie do pracy z polskim słownikiem semantycznym. 
Pozwala na odczyt z bazy/grafów, zarządzanie elementami, wyszukiwanie 
i eksportowanie danych.
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


class _SetupConfiguration:
    REPO_URL = "https://github.com/radlab-dev-group/radlab-plwordnet"

    @dataclass
    class SetupConfig:
        module_name: str = "plwordnet-handler"
        module_version: str = "1.5.1"
        handler_module_name: str = "plwordnet_handler"
        short_description: str = "Narzędzie do pracy ze Słowosiecią"

        author: str = "RadLab Team"
        author_email: str = "pawel@radlab.dev"

        radlab_site: str = "https://radlab.dev"
        radlab_resources: str = (
            "https://resources.radlab.dev/persistant/plwn_handler"
        )
        remote_base_resources_dir: str = "resources"
        remote_base_graphs_dir: str = "plwordnet/graphs"
        remote_test_graph_name: str = "slowosiec_test.gz"
        remote_full_graph_name: str = "slowosiec_full.gz"

        @property
        def full_plwn_graph_url(self):
            return (
                f"{self.radlab_resources}/"
                f"{self.remote_base_resources_dir}/"
                f"{self.remote_base_graphs_dir}/"
                f"{self.remote_full_graph_name}"
            )

        @property
        def test_plwn_graph_url(self):
            return (
                f"{self.radlab_resources}/"
                f"{self.remote_base_resources_dir}/"
                f"{self.remote_base_graphs_dir}/"
                f"{self.remote_test_graph_name}"
            )


class _ReqReadmeHandler:
    @staticmethod
    def read_requirements():
        requirements_path = os.path.join(
            os.path.dirname(__file__), "requirements.txt"
        )

        if os.path.exists(requirements_path):
            with open(requirements_path, "r", encoding="utf-8") as f:
                return [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        return []

    @staticmethod
    def read_long_description():
        readme_files = ["README.md"]
        for readme_file in readme_files:
            readme_path = os.path.join(os.path.dirname(__file__), readme_file)
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    return f.read()
        return __SHORT__DESCRIPTION__


class _ResourcesHandler:
    """
    Handler for collecting and installing resources directory.
    """

    base_package_data = {
        "": ["*.txt", "*.md", "*.rst"],
        _SetupConfiguration.SetupConfig.handler_module_name: [
            "resources/*",
            "resources/**/*",
        ],
    }

    @staticmethod
    def get_resources_files():
        """
        Recursively collect all files in the resources directory.

        Returns:
            Dictionary mapping package paths to file lists
        """
        resources_dir = Path("resources")

        if not resources_dir.exists():
            print("Warning: resources directory not found in the repository")
            return {}

        package_data = {}
        for root, dirs, files in os.walk(resources_dir):
            if not files:
                continue

            rel_root = Path(root).relative_to(resources_dir)
            if str(rel_root) == ".":
                file_patterns = [f"resources/{f}" for f in files]
            else:
                file_patterns = [f"resources/{rel_root}/{f}" for f in files]

            package_path = _SetupConfiguration.SetupConfig.handler_module_name
            if package_path not in package_data:
                package_data[package_path] = []
            package_data[package_path].extend(file_patterns)

        return package_data

    @staticmethod
    def copy_resources_to_package():
        """
        Copy resources directory to plwordnet_handler package.
        This ensures resources are included in the package structure.
        """
        source_resources = Path("resources")
        target_resources = Path("plwordnet_handler/resources")

        if not source_resources.exists():
            print("Warning: Source resources directory not found")
            return False

        try:
            # Remove the existing target if it exists
            if target_resources.exists():
                shutil.rmtree(target_resources)

            # Copy the entire resources directory
            shutil.copytree(source_resources, target_resources)
            print(
                f"Successfully copied resources from "
                f"{source_resources} to {target_resources}"
            )

            # Create __init__.py in the resources directory to make it a package
            init_file = target_resources / "__init__.py"
            init_file.write_text('"""PLWordNet resources package."""\n')

            return True
        except Exception as e:
            print(f"Error copying resources: {e}")
            return False


class _GraphDownloadMixin:
    """
    Mixin for downloading graphs during installation.
    """

    ENV_DOWN_TEST_PLWN = "PLWORDNET_DOWNLOAD_TEST"
    ENV_DOWN_FULL_PLWN = "PLWORDNET_DOWNLOAD_FULL"

    def download_graphs_if_requested(self):
        """
        Download graphs if environment variables are set.
        """
        config = _SetupConfiguration.SetupConfig()

        download_test = os.environ.get(self.ENV_DOWN_TEST_PLWN, "").lower() in (
            "1",
            "true",
            "yes",
        )
        download_full = os.environ.get(self.ENV_DOWN_FULL_PLWN, "").lower() in (
            "1",
            "true",
            "yes",
        )

        if not (download_test or download_full):
            print("No graph download requested.")
            return

        # Determine target directory
        if hasattr(self, "install_lib") and self.install_lib:
            target_dir = self.install_lib
        else:
            # Fallback for develop mode
            import site

            target_dir = site.getsitepackages()[0]

        graphs_dir = os.path.join(
            target_dir, "plwordnet_handler", "resources", "graphs"
        )
        os.makedirs(graphs_dir, exist_ok=True)
        print(f"Target graphs directory: {graphs_dir}")

        if download_test:
            print("Downloading test graph data...")
            self._download_and_extract_graph(
                config.test_plwn_graph_url,
                graphs_dir,
                "plwn_test_graph",
            )

        if download_full:
            print("Downloading full graph data...")
            self._download_and_extract_graph(
                config.full_plwn_graph_url,
                graphs_dir,
                "plwn_full_graph",
            )

    @staticmethod
    def _download_and_extract_graph(url, destination_path, graph_type):
        """
        Download and extract a gzipped graph file.
        """
        archive_path = graph_type + ".gz"
        try:
            print(f"Downloading {graph_type} from {url}...")
            urllib.request.urlretrieve(url, archive_path)

            print(f"Extracting {graph_type}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=destination_path)

            if os.path.exists(archive_path):
                os.remove(archive_path)

            print(f"Successfully installed {graph_type} to {destination_path}")
        except Exception as e:
            print(f"Error downloading {graph_type}: {e}")
            for path in [archive_path, destination_path]:
                if os.path.exists(path):
                    os.remove(path)


class _CustomInstallCommand(install, _GraphDownloadMixin):
    """
    Custom installation command that copies resources and downloads graphs.
    """

    def run(self):
        # Copy resources before installation
        print("Copying resources directory to package...")
        _ResourcesHandler.copy_resources_to_package()

        # Run normal installation
        install.run(self)

        # Download graphs if requested
        self.download_graphs_if_requested()


class _CustomDevelopCommand(develop, _GraphDownloadMixin):
    """Custom develop command that copies resources and downloads graphs."""

    def run(self):
        # Copy resources before development installation
        print("Copying resources directory to package...")
        _ResourcesHandler.copy_resources_to_package()

        # Run normal develop installation
        develop.run(self)

        # Download graphs if requested
        self.download_graphs_if_requested()


class _CustomEggInfoCommand(egg_info, _GraphDownloadMixin):
    """Custom egg_info command that ensures resources are available."""

    def run(self):
        # Copy resources before egg_info
        print("Ensuring resources are available...")
        _ResourcesHandler.copy_resources_to_package()

        # Run normal egg_info
        egg_info.run(self)

        # Download graphs if explicitly requested during egg_info
        if any(
            env in os.environ
            for env in [self.ENV_DOWN_TEST_PLWN, self.ENV_DOWN_FULL_PLWN]
        ):
            try:
                self.download_graphs_if_requested()
            except Exception as e:
                print(f"Warning: Could not download graphs during egg_info: {e}")


class _ResourceInit:
    """
    Mixin for downloading graphs during installation.
    """

    @staticmethod
    def install_resources():
        print("Preparing resources for installation...")
        _ResourcesHandler.copy_resources_to_package()
        resources_package_data = _ResourcesHandler.get_resources_files()
        for package, files in resources_package_data.items():
            if package in _ResourcesHandler.base_package_data:
                _ResourcesHandler.base_package_data[package].extend(files)
            else:
                _ResourcesHandler.base_package_data[package] = files


_ResourceInit.install_resources()


setup(
    name=_SetupConfiguration.SetupConfig.module_name,
    version=_SetupConfiguration.SetupConfig.module_version,
    description=_SetupConfiguration.SetupConfig.short_description,
    long_description=_ReqReadmeHandler.read_long_description(),
    long_description_content_type="text/markdown",
    author=_SetupConfiguration.SetupConfig.author,
    author_email=_SetupConfiguration.SetupConfig.author_email,
    url=_SetupConfiguration.REPO_URL,
    packages=find_packages(),
    include_package_data=True,
    package_data=_ResourcesHandler.base_package_data,
    install_requires=_ReqReadmeHandler.read_requirements(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "plwordnet-cli=apps.cli.plwordnet_cli:main",
            "plwordnet-milvus=apps.cli.plwordnet_milvus_cli:main",
        ],
    },
    keywords="nlp, wordnet, polish, linguistics, semantic",
    zip_safe=False,
    cmdclass={
        "install": _CustomInstallCommand,
        "develop": _CustomDevelopCommand,
        "egg_info": _CustomEggInfoCommand,
    },
)
