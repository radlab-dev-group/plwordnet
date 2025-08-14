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

    def __init__(self, milvus: MilvusWordNetInsertHandler, batch_size: int = 1000):
        """
        Initialize the embedding consumer with a Milvus insert handler.

        Args:
            milvus: MilvusWordNetInsertHandler instance for database operations
            batch_size: Batch size for inserting embeddings (default 1000)
        """

        self.milvus = milvus
        self.milvus.connect()

        self.batch_size = batch_size

        self._batch_lu = []
        self._batch_syn = []
        self._batch_lu_e = []

    def add_embedding(
        self, embedding_dict: Dict[str, Any], model_name: str, batch_size: int = None
    ) -> bool:
        """
        Add an embedding to the appropriate Milvus collection based on type.

        Args:
            embedding_dict: Dictionary containing embedding data with
            type information
            model_name: Name of the model used to generate the embedding
            batch_size: Batch size for inserting embeddings (default None,
            self.batch_size will be used)

        Returns:
            bool: True if insertion is successful, False otherwise

        Raises:
            NotImplementedError: If the embedding type is not supported
        """
        if not len(embedding_dict):
            return False

        emb_type = embedding_dict.get("type", None)
        if batch_size is None:
            batch_size = self.batch_size

        if emb_type == "lu_example":
            self.__process_lu_example(
                embedding_dict=embedding_dict,
                model_name=model_name,
                batch_size=batch_size,
            )
        elif emb_type == "lu":
            self.__process_lu(
                embedding_dict=embedding_dict,
                model_name=model_name,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")
        return True

    def flush(self):
        """
        Flush all batched embeddings to Milvus collections.

        Processes and inserts any remaining embeddings in the internal batches
        for lexical unit examples, lexical units, and synsets. Clears the batch
        buffers after successful insertion.

        Raises:
            NotImplementedError: Method implementation is incomplete
        """
        if len(self._batch_lu_e):
            self.milvus.insert_lu_examples_embeddings(
                data=self._batch_lu_e, batch_size=self.batch_size
            )
            self._batch_lu_e = []

        if len(self._batch_lu):
            self.milvus.insert_lu_embeddings(
                data=self._batch_lu, batch_size=self.batch_size
            )
            self._batch_lu = []

        if len(self._batch_syn):
            self._batch_syn = []
            raise NotImplementedError("Synset flushing not implemented")

    def __process_lu_example(
        self, embedding_dict: Dict[str, Any], model_name: str, batch_size: int
    ):
        """
        Process and insert a lexical unit example embedding.

        Args:
            embedding_dict: Dictionary containing lexical unit example data
            model_name: Name of the model used to generate the embedding
        """
        lu = embedding_dict["lu"]
        text = embedding_dict["texts"]
        embedding = embedding_dict["embedding"].cpu().numpy()
        if not len(text) or not len(text[0].strip()):
            return
        text = text[0].strip()

        item = {
            "id": lu.ID,
            "embedding": embedding,
            "example": text,
            "model_name": model_name,
        }
        self._batch_lu_e.append(item)

        if len(self._batch_lu_e) >= batch_size:
            _inserted = self.milvus.insert_lu_examples_embeddings(
                data=self._batch_lu_e, batch_size=len(self._batch_lu_e)
            )
            self._batch_lu_e = []

    def __process_lu(
        self, embedding_dict: Dict[str, Any], model_name: str, batch_size: int
    ):
        """
        Process and insert a lexical unit embedding.

        Args:
            embedding_dict: Dictionary containing lexical unit data
            model_name: Name of the model used to generate the embedding
        """
        lu = embedding_dict["lu"]
        embedding = embedding_dict["embedding"].cpu().numpy()

        item = {
            "id": lu.ID,
            "embedding": embedding,
            "lemma": lu.lemma,
            "pos": lu.pos,
            "domain": lu.domain,
            "variant": lu.variant,
            "model_name": model_name,
        }
        self._batch_lu.append(item)

        if len(self._batch_lu) >= batch_size:
            _inserted = self.milvus.insert_lu_embeddings(
                data=self._batch_lu, batch_size=len(self._batch_lu)
            )
            self._batch_lu = []
