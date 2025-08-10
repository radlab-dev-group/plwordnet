"""
Example usage documentation for the Polish Wordnet CLI application.
"""

from plwordnet_handler.cli.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_NX_OUT_DIR,
    DEFAULT_NX_GRAPHS_DIR,
    DEFAULT_DB_CFG_PATH,
)


EXAMPLE_USAGE = f"""
Example usage:

# Load from NetworkX graphs (default):
python plwordnet-cli \\
        --nx-graph-dir {DEFAULT_NX_GRAPHS_DIR} \\
        --extract-wikipedia-articles \\
        --log-level {DEFAULT_LOG_LEVEL}

# Convert from database to NetworkX graphs:
python plwordnet-cli \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --convert-to-nx-graph \\
        --nx-graph-dir {DEFAULT_NX_OUT_DIR} \\
        --log-level {DEFAULT_LOG_LEVEL}

# Load from database directly:
python plwordnet-cli \\
        --use-database \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --log-level {DEFAULT_LOG_LEVEL}
"""
