import torch

from typing import Dict, Any


class EmbeddingMilvusConsumer:
    def __init__(self, milvus):
        self.milvus = milvus

    def add_embedding(self, embedding_dict: Dict[str, Any]):
        lu = embedding_dict.get("lu", None)
        texts = embedding_dict.get("texts", [])
        embedding = embedding_dict.get("embedding", None)
        emb_type = embedding_dict.get("type", None)
        strategy = embedding_dict.get("strategy", None)

        if emb_type == "lu_example":
            pass
        elif emb_type == "lu":
            pass
        else:
            raise NotImplementedError
