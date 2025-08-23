import os
import json
from typing import List, Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.elems.lu import LexicalUnit
from plwordnet_handler.base.structure.elems.rel_type import RelationType
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet


class RelGATDatasetIdentifiersAligner:
    """
    Based on the available lu/rels prepare the mapping original id of lu/rel
    to new idx enumerated from 0 to the number of used lu/relations
    """

    def __init__(
        self,
        plwn_api: Optional[PolishWordnet] = None,
        prepare_mapping: bool = False,
        mapping_path: Optional[str] = None,
        log_level: Optional[str] = "INFO",
        logger_file_name: Optional[str] = None,
    ):
        self.plwn_api = plwn_api
        self.mapping_path = mapping_path

        # ---------------------------------------------------------------------
        #   - mapping of aligned lexical units identifiers to original ID
        self._lu_align_to_original = {}  # aligned lu id
        #   - reversed mapping; aligned lu id to original LU identifier
        self._lu_original_to_align = {}

        # ---------------------------------------------------------------------
        #   - mapping of aligned rel identifier to original identifier
        self._rel_align_to_original = {}
        #   - reverse mapping of original relation identifier to aligned identifier
        self._rel_original_to_align = {}
        #   - mapping the relation name (string) to aligned relation identifier
        self._rel_name_to_aligned_id = {}
        #   - mapping aligned relation id to relation name (string)
        self._aligned_id_to_rel_name = {}

        self.logger = prepare_logger(
            logger_name=__name__,
            log_level=log_level,
            logger_file_name=logger_file_name,
        )

        self._filename_to_mapping = {
            "lu_align_to_original.json": self._lu_align_to_original,
            "lu_original_to_align.json": self._lu_original_to_align,
            "rel_align_to_original.json": self._rel_align_to_original,
            "rel_original_to_align.json": self._rel_original_to_align,
            "rel_name_to_aligned_id.json": self._rel_name_to_aligned_id,
            "aligned_id_to_rel_name.json": self._aligned_id_to_rel_name,
        }

        if prepare_mapping:
            if plwn_api is None:
                raise Exception("PLWN api is required when prepare_mapping is True!")

            self.logger.warning(
                "RelGAT dataset identifier is created with prepare_mapping=True. "
                "Aligned dataset will be prepared, pleas wait... "
                "If you have prepared mapping, you should to call the "
                "RelGATDatasetIdentifiersAligner with default behaviour "
                "with option prepare_mapping=False. If you are actually "
                "preparing mapping please ignore this message."
            )
            self._prepare_mapping()
        else:
            if mapping_path is None:
                raise Exception(
                    "Path with mappings is required to use "
                    "RelGATDatasetIdentifiersAligner "
                    "in mapping mode (prepare_mapping=False)"
                )
            self.logger.debug(f"Loading RelGAT mapping from {mapping_path}")
            self._load_mappings_from_dir(mapping_path=mapping_path)

    def export_to_dir(self, out_directory: str) -> None:
        os.makedirs(out_directory, exist_ok=True)

        for f_name, data in self._filename_to_mapping.items():
            _m_name = f_name.replace(".json", "")
            out_file_path = os.path.join(out_directory, f_name)
            self.logger.info(f"Exporting mapping {_m_name} to {out_file_path}")
            with open(out_file_path, "w", encoding="utf-8") as f:
                if type(data) in [list, dict]:
                    data = json.dumps(data, indent=2, ensure_ascii=False)
                f.write(data)

    def _prepare_mapping(self) -> None:
        self.logger.info("Preparing relations and lexical units mapping")
        self._prepare_relations()
        self._prepare_lexical_units()

    def _load_mappings_from_dir(self, mapping_path: str) -> None:
        self.logger.info(f"Loading mappings from path {mapping_path}")
        for f_name in self._filename_to_mapping.keys():
            _m_name = f_name.replace(".json", "")
            file_path = os.path.join(mapping_path, f_name)

            if not os.path.exists(file_path):
                self.logger.error(f"Mapping file {f_name} not found at {file_path}")
                raise Exception(f"Mapping file {file_path} does not exist!")

            self.logger.debug(f"Loading mapping {_m_name} from {file_path}")
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    self._filename_to_mapping[f_name] = json.load(f)
            else:
                self.logger.error(f"Invalid file format: {file_path}")
                raise Exception(f"Invalid file format: ...{file_path[-4:]}")

    def _prepare_relations(self):
        self.logger.info(" - preparing relations mappings")
        self.__align_rel_identifiers(rels_list=self.plwn_api.get_relation_types())

    def _prepare_lexical_units(self):
        self.logger.info(" - preparing lexical units mappings")
        self.__align_lu_identifiers(lu_list=self.plwn_api.get_lexical_units())

    def __align_rel_identifiers(self, rels_list: List[RelationType]) -> None:
        self.logger.info(
            f"   - number of relations to align identifiers {len(rels_list)}"
        )

        self._rel_align_to_original.clear()
        self._rel_original_to_align.clear()
        self._rel_name_to_aligned_id.clear()
        self._aligned_id_to_rel_name.clear()
        for idx, rel in enumerate(rels_list):
            self._rel_align_to_original[idx] = rel.ID
            self._rel_original_to_align[rel.ID] = idx
            self._rel_name_to_aligned_id[rel.name] = idx
            self._aligned_id_to_rel_name[idx] = rel.name

        self.logger.info(
            f"   - aligned rel -> original: {len(self._rel_align_to_original)}"
        )
        self.logger.info(
            f"   - original -> aligned rel: {len(self._rel_original_to_align)}"
        )
        self.logger.info(
            f"   - relation name -> aligned id: {len(self._rel_name_to_aligned_id)}"
        )
        self.logger.info(
            f"   - aligned id -> relation name: {len(self._aligned_id_to_rel_name)}"
        )

    def __align_lu_identifiers(self, lu_list: List[LexicalUnit]) -> None:
        self.logger.info(
            f"   - number of lexical units to align identifiers {len(lu_list)}"
        )
        self._lu_align_to_original.clear()
        self._lu_original_to_align.clear()
        for idx, lu in enumerate(lu_list):
            self._lu_align_to_original[idx] = lu.ID
            self._lu_original_to_align[lu.ID] = idx

        self.logger.info(
            f"   - aligned lu -> original: {len(self._lu_align_to_original)}"
        )
        self.logger.info(
            f"   - original -> aligned lu: {len(self._lu_original_to_align)}"
        )
