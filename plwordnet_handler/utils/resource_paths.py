import pkg_resources

from pathlib import Path
from typing import Optional

from plwordnet_handler.base.connectors.connector_data import GraphMapperData


class ResourcePaths:
    """
    Helper class for managing installed resource paths.
    """

    DEFAULT_PLWN_DB_CONFIG = "plwordnet-mysql-db.json"

    PACKAGE_NAME = "plwordnet_handler"
    RESOURCES_SUBDIR = "resources"
    GRAPHS_SUBDIR = f"{RESOURCES_SUBDIR}/graphs"
    FULL_GRAPHS_SUBDIR = f"{GRAPHS_SUBDIR}/plwordnet_full/nx/graphs"
    TEST_GRAPHS_SUBDIR = f"{GRAPHS_SUBDIR}/plwordnet_test/nx/graphs"

    # Graph file names/resources names (installed by setup.py)
    GRAPH_RESOURCES_FILES = [
        GraphMapperData.GRAPH_TYPES[GraphMapperData.G_LU],
        GraphMapperData.GRAPH_TYPES[GraphMapperData.G_SYN],
        GraphMapperData.RELATION_TYPES_FILE,
        GraphMapperData.LU_IN_SYNSET_FILE,
    ]

    @classmethod
    def get_installed_graphs_dir(cls, graph_type) -> Optional[Path]:
        """
        Get the directory path where graphs are installed.

        Returns:
            Path to installed graphs directory or None if not found
        """
        graph_dir = cls.FULL_GRAPHS_SUBDIR
        if "test" in graph_type:
            graph_dir = cls.TEST_GRAPHS_SUBDIR
        try:
            # Try to get the resource path using pkg_resources
            graphs_dir = pkg_resources.resource_filename(cls.PACKAGE_NAME, graph_dir)
            graphs_path = Path(graphs_dir)
            if graphs_path.exists():
                return graphs_path
        except (pkg_resources.DistributionNotFound, FileNotFoundError):
            pass

        # Fallback: try to find relative to the current module
        try:
            current_dir = Path(__file__).parent.parent
            graphs_path = current_dir / "resources" / "graphs"
            if graphs_path.exists():
                return graphs_path
        except Exception:
            pass
        return None

    @classmethod
    def get_installed_resources_dir(cls) -> Optional[Path]:
        """
        Get the directory path where resources are installed.

        Returns:
            Path to installed resources directory or None if not found
        """
        try:
            # Try to get the resource path using pkg_resources
            graphs_dir = pkg_resources.resource_filename(
                cls.PACKAGE_NAME, cls.RESOURCES_SUBDIR
            )
            resources_path = Path(graphs_dir)
            if resources_path.exists():
                return resources_path
        except (pkg_resources.DistributionNotFound, FileNotFoundError):
            pass

        # Fallback: try to find relative to the current module
        try:
            current_dir = Path(__file__).parent.parent
            resources_path = current_dir / cls.RESOURCES_SUBDIR
            if resources_path.exists():
                return resources_path
        except Exception:
            pass
        return None

    @classmethod
    def get_installed_graph_resources_path(cls, graph_type: str) -> Optional[Path]:
        """
        Get a path to a specific installed graph (or resource) file.

        Args:
            graph_type: Graph type: `full` or `test`

        Returns:
            Path to a graph dir or None if not found graphs
        """

        graphs_dir = cls.get_installed_graphs_dir(graph_type=graph_type)
        if graphs_dir is None:
            return None

        for g_file in cls.GRAPH_RESOURCES_FILES:
            graph_file = graphs_dir / g_file
            if not graph_file.exists():
                print(f"Graph/resource file does not exist: {graph_file}")
                return None
        return graphs_dir

    @classmethod
    def get_installed_resources_path(cls) -> Optional[Path]:
        """
        Get a path to a specific installed graph file.

        Returns:
            Path to a resource or None if not found
        """

        resources_dir = cls.get_installed_resources_dir()
        if resources_dir is None:
            return None
        return resources_dir

    @classmethod
    def get_default_db_config(cls) -> Optional[Path]:
        res_path = cls.get_installed_resources_path()
        if res_path is None:
            return None
        return res_path / cls.DEFAULT_PLWN_DB_CONFIG


def get_default_graph_path(prefer_full: bool = True) -> Optional[str]:
    """
    Get a default graph path, preferring full to test if available.

    Args:
        prefer_full: If True, prefer full graph over test graph

    Returns:
        Path to graph directory as string, or None if no graphs are found
    """
    graph_type = "test"
    if prefer_full:
        graph_type = "full"
    _gp = ResourcePaths.get_installed_graph_resources_path(graph_type=graph_type)
    if _gp is None and graph_type != "test":
        _gp = ResourcePaths.get_installed_graph_resources_path(graph_type="test")
        if _gp is not None:
            print()
            print("\t.===========================================================.")
            print("\t|                                                           |")
            print("\t|   [INFO] Instead of full, test graph is available.        |")
            print("\t|   [INFO] To use full graph reinstall package with option: |")
            print("\t|                                                           |")
            print("\t|      export PLWORDNET_DOWNLOAD_TEST=1                     |")
            print("\t|      export PLWORDNET_DOWNLOAD_FULL=1                     |")
            print("\t|      pip install .                                        |")
            print("\t|                                                           |")
            print("\t'===========================================================.")
            print()
    return _gp


def get_default_db_config_path() -> Optional[Path]:
    """
    Get a default db config path.

    Returns:
        Path to a default mysql connection config file
    """
    return ResourcePaths.get_default_db_config()
