from pathlib import Path
from typing import Union

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)
from plwordnet_ml.dataset.aligned_id.aligned_dataset_id import (
    RelGATDatasetIdentifiersAligner,
)


class RelGATExporter:
    def __init__(
        self,
        plwn_api: PolishWordnet,
        milvus_handler: MilvusWordNetSearchHandler,
        out_directory: str,
    ):
        self.plwn_api = plwn_api
        self.milvus_handler = milvus_handler
        self.out_directory = self._ensure_dir(path=out_directory)

        self._lu_list = []
        self._rels_list = []
        self._lu_rels_list = []
        self._units_and_synsets = []
        self.logger = prepare_logger(
            logger_name=__name__,
            log_level=self.plwn_api.log_level,
            logger_file_name=self.plwn_api.log_file_name,
        )

    def export(self):
        self.logger.info("Exporting RelGAT dataset")

        data_aligner = RelGATDatasetIdentifiersAligner(
            plwn_api=self.plwn_api,
            prepare_mapping=True,
        )

        raise NotImplementedError("Not implemented yet")

    @staticmethod
    def _ensure_dir(path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
