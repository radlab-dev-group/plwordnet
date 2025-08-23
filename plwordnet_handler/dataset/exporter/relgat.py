import os
import pickle

import tqdm
import json

from typing import Optional, List

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.elems.rel_type import SYNONYMY_ID
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)
from plwordnet_ml.dataset.aligned_id.aligned_dataset_id import (
    RelGATDatasetIdentifiersAligner,
)
from plwordnet_handler.base.structure.elems.lu_in_synset import (
    LexicalUnitAndSynsetFakeRelation,
)


class RelGATExporter:
    LU_EMBEDDING_FILENAME = "lexical_units_embedding.pickle"
    RELATION_MAPPING_FILENAME = "relation_to_idx.json"
    ALL_RELATIONS_TRIPLETS = "relations_triplets.json"

    def __init__(
        self,
        plwn_api: PolishWordnet,
        milvus_handler: MilvusWordNetSearchHandler,
        aligner: RelGATDatasetIdentifiersAligner,
        out_directory: Optional[str] = None,
        accept_pos: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ):
        self.limit = limit

        self.aligner = aligner
        self.plwn_api = plwn_api
        self.accept_pos = accept_pos
        self.milvus_handler = milvus_handler
        self.out_directory = out_directory

        self._lu_to_emb = {}
        self._rel_to_idx = {}
        self._relations = []  # (src_idx, dst_idx, rel_name)

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

        os.makedirs(out_directory, exist_ok=True)

        self._prepare_data()
        self._export_data_to_dir(out_directory=out_directory)

    def _prepare_data(self):
        self.logger.info("Preparing data to export")
        self._prepare_rel_to_idx()
        self._prepare_relations()
        self._prepare_embeddings(limit=self.limit)

    def _export_data_to_dir(self, out_directory: str) -> None:
        self.logger.info(f"Storing RelGAT mappings to {out_directory}")

        data_export = [
            (self.LU_EMBEDDING_FILENAME, self._lu_to_emb),
            (self.RELATION_MAPPING_FILENAME, self._rel_to_idx),
            (self.ALL_RELATIONS_TRIPLETS, self._relations),
        ]
        for f_name, data in data_export:
            out_file = os.path.join(out_directory, f_name)
            self.logger.info(f" - exporting {out_file}")

            if f_name.endswith(".json"):
                with open(out_file, "w") as f:
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
            elif f_name.endswith(".pickle"):
                with open(out_file, "wb") as f:
                    pickle.dump(data, f)
            else:
                raise NotImplementedError(
                    f"{f_name} is not supported file. "
                    f"Only [json, pickle] are supported!"
                )

    def _prepare_rel_to_idx(self) -> None:
        self.logger.info(" - preparing relation name to aligned id mapping")
        self._rel_to_idx = self.aligner.rel_name_to_aligned_idx_dict.copy()

    def _prepare_relations(self) -> None:
        self.logger.info(
            " - preparing relation triplets (src_idx, dst_idx, rel_name)"
        )

        lu_relations = self._prepare_relations_from_list(
            rel_list=self.plwn_api.get_lexical_relations()
        )
        self.logger.info(f"   - lexical units: {len(lu_relations)}")

        syn_rels = self._prepare_relations_from_list(
            rel_list=self._synonymy_as_relations()
        )
        self.logger.info(f"   - synonyms: {len(syn_rels)}")

        self._relations = lu_relations + syn_rels
        self.logger.info(f"   - all relations: {len(self._relations)}")

    def _prepare_embeddings(self, limit: Optional[int] = None) -> None:
        self.logger.info(" - preparing embeddings for each lexical unit")
        self._lu_to_emb = {}

        all_lexical_units = self.plwn_api.get_lexical_units()
        if self.accept_pos is not None and len(self.accept_pos):
            all_lexical_units = [
                lu for lu in all_lexical_units if lu.pos in self.accept_pos
            ]
        self.logger.info(f"  -> number of lexical units: {len(all_lexical_units)}")
        if limit is not None and limit > 0:
            all_lexical_units = all_lexical_units[:limit]
            self.logger.info(
                f"  -> number of LU after limit: {len(all_lexical_units)}"
            )

        with tqdm.tqdm(
            total=len(all_lexical_units), desc="Retrieving embeddings from Milvus"
        ) as pbar:
            for lu in all_lexical_units:
                lu_emb = self.milvus_handler.get_lexical_unit_embedding(lu_id=lu.ID)
                if not lu_emb:
                    continue
                a_lu_id = self.aligner.aligned_lexical_unit_id(orig_lu_id=lu.ID)
                if a_lu_id is None:
                    continue

                self._lu_to_emb[a_lu_id] = lu_emb["embedding"]

                if limit is not None and limit > 0:
                    if len(self._lu_to_emb) >= limit:
                        break

                pbar.update(1)

    def _synonymy_as_relations(self) -> List[LexicalUnitAndSynsetFakeRelation]:
        synonymy_rels = []
        _uas = self.plwn_api.get_units_and_synsets(return_mapping=True)
        for s_id, lu_list in _uas.items():
            if len(lu_list) < 2:
                # Skip if a synset has a single lexical unit
                continue

            lu_list = list(lu_list)
            for _p in range(len(lu_list)):
                for _ch in range(_p, len(lu_list)):
                    if _p == _ch:
                        # Skip self-synonymy link
                        continue

                    synonymy_rels.append(
                        LexicalUnitAndSynsetFakeRelation(
                            PARENT_ID=lu_list[_p],
                            CHILD_ID=lu_list[_ch],
                            REL_ID=SYNONYMY_ID,
                        )
                    )
        return synonymy_rels

    def _prepare_relations_from_list(self, rel_list) -> List:
        lu_relations = []
        for rel in rel_list:
            p_id = self.aligner.aligned_lexical_unit_id(orig_lu_id=rel.PARENT_ID)
            if p_id is None:
                self.logger.warning(
                    f"Cannot find mapping for LU {rel.PARENT_ID}. Skipping relation."
                )
                continue

            ch_id = self.aligner.aligned_lexical_unit_id(orig_lu_id=rel.CHILD_ID)
            if ch_id is None:
                self.logger.warning(
                    f"Cannot find mapping for LU {rel.CHILD_ID}. Skipping relation."
                )
                continue

            rel_name = self.aligner.aligned_relation_name(orig_rel_id=rel.REL_ID)
            if rel_name is None:
                self.logger.warning(
                    f"Cannot find mapping for relation {rel.REL_ID}. Skipping."
                )
                continue

            lu_relations.append([p_id, ch_id, rel_name])
        return lu_relations
