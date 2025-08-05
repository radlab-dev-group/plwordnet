class GraphMapperData:
    """
    Configuration data for mapping operations.

    This class defines constants and mappings used for converting database data
    into NetworkX graph formats. It specifies the available graph types and their
    corresponding file storage configurations.

    Class Attributes:
        G_UAS (str): Identifier for units and synsets graph type.
        G_LU (str): Identifier for a lexical units graph type.
        G_SYN (str): Identifier for a synsets graph type.
        GRAPH_TYPES (dict): Mapping of graph type identifiers
                            to their pickle file names.
        GRAPH_DIR (str): Directory path where graph files are stored.
    """

    G_UAS = "units_and_synsets"
    G_LU = "lexical_units"
    G_SYN = "synsets"

    GRAPH_TYPES = {
        G_UAS: f"{G_UAS}.pickle",
        G_LU: f"{G_LU}.pickle",
        G_SYN: f"{G_SYN}.pickle",
    }

    GRAPH_DIR = "nx/graphs"
