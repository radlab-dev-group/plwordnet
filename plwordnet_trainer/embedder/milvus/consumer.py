from typing import Dict, Any

from plwordnet_handler.base.connectors.milvus.core.io import (
    MilvusWordNetInsertHandler,
)


class EmbeddingMilvusConsumer:
    """
    Consumer class for inserting embeddings into Milvus collections.

    Handles the processing and insertion of different types of embeddings
    (lexical units, synsets, and examples) into appropriate Milvus collections.
    """

    def __init__(self, milvus: MilvusWordNetInsertHandler):
        """
        Initialize the embedding consumer with a Milvus insert handler.

        Args:
            milvus: MilvusWordNetInsertHandler instance for database operations
        """

        self.milvus = milvus
        self.milvus.connect()

    def add_embedding(self, embedding_dict: Dict[str, Any], model_name: str) -> bool:
        """
        Add an embedding to the appropriate Milvus collection based on type.

        Args:
            embedding_dict: Dictionary containing embedding data with
            type information
            model_name: Name of the model used to generate the embedding

        Returns:
            bool: True if insertion is successful, False otherwise

        Raises:
            NotImplementedError: If the embedding type is not supported
        """
        emb_type = embedding_dict.get("type", None)

        if emb_type == "lu_example":
            return self.__process_lu_example(
                embedding_dict=embedding_dict, model_name=model_name
            )
        elif emb_type == "lu":
            return self.__process_lu(
                embedding_dict=embedding_dict, model_name=model_name
            )
        else:
            raise NotImplementedError(f"Unknown embedding type: {emb_type}")

    def __process_lu_example(
        self, embedding_dict: Dict[str, Any], model_name: str
    ) -> bool:
        """
        Process and insert a lexical unit example embedding.

        Args:
            embedding_dict: Dictionary containing lexical unit example data
            model_name: Name of the model used to generate the embedding

        Returns:
            bool: True if insertion is successful, False if text is empty
        """
        lu = embedding_dict["lu"]
        text = embedding_dict["texts"]
        if not len(text) or not len(text[0].strip()):
            return False
        text = text[0].strip()

        return self.milvus.insert_single_lu_example(
            lu_id=lu.ID,
            embedding=embedding_dict["embedding"],
            example=text,
            model_name=model_name,
        )

    def __process_lu(self, embedding_dict: Dict[str, Any], model_name: str) -> bool:
        """
        Process and insert a lexical unit embedding.

        Args:
            embedding_dict: Dictionary containing lexical unit data
            model_name: Name of the model used to generate the embedding

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        lu = embedding_dict["lu"]
        return self.milvus.insert_single_lu(
            lu_id=lu.ID,
            embedding=embedding_dict["embedding"],
            lemma=lu.lemma,
            pos=lu.pos,
            domain=lu.domain,
            variant=lu.variant,
            model_name=model_name,
        )
