class EmbeddingTypes:
    """
    Defines different types of embeddings
    """

    class Base:
        """
        Base embeddings, created from Lexical unit examples and synsets using
        only embeddings from lexical units examples usage, definition, etc.
        """

        lu = "base_lu"
        lu_fake = "base_lu_fake"
        lu_example = "base_lu_example"
        synset = "base_synset"
