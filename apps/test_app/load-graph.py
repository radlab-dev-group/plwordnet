import pickle


def load_graph_and_print_nodes(g_path):
    """
    Loads a NetworkX graph from a pickle file using the standard pickle module
    and prints its nodes.

    Args:
        g_path (str): The path to the pickle file containing the graph.
    """
    try:
        with open(g_path, "rb") as f:  # Open the file in binary read mode ('rb')
            graph = pickle.load(f)
        print("Nodes in the graph:")
        print(list(graph.nodes(data=True)))
    except FileNotFoundError:
        print(f"Error: File not found at path: {g_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


load_graph_and_print_nodes("graphs/lexical_units.pickle")
