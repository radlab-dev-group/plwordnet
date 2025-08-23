from typing import Optional

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
        self.out_directory = out_directory

        self._lu_list = []
        self._rels_list = []
        self._lu_rels_list = []
        self._units_and_synsets = []
        self.logger = prepare_logger(
            logger_name=__name__,
            log_level=self.milvus_handler.log_level,
            logger_file_name=self.milvus_handler.logger_name,
        )

    def export_to_dir(self, out_directory: Optional[str] = None) -> None:
        self.logger.info("Exporting RelGAT mappings")
        if out_directory is None:
            out_directory = self.out_directory
            if out_directory is None:
                raise TypeError(
                    "out_directory is required to export RelGAT mappings"
                )

        data_aligner = RelGATDatasetIdentifiersAligner(
            plwn_api=self.plwn_api,
            prepare_mapping=True,
            log_level=self.milvus_handler.log_level,
            logger_file_name=self.milvus_handler.logger_name,
        )
        data_aligner.export_to_dir(out_dir=out_directory)
