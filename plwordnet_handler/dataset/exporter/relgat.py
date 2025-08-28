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
        self._prepare_embeddings(limit=self.limit)
        self._prepare_relations()

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

    def _prepare_relations(self) -> None:
        self.logger.info(
            " - preparing relation triplets (src_idx, dst_idx, rel_name)"
        )

        lu_relations_clear = self._prepare_relations_from_list(
            rel_list=self.plwn_api.get_lexical_relations(), check_embedding=True
        )
        self.logger.info(
            f"   - relations with both embeddings : {len(lu_relations_clear)}"
        )

        synonymy_name = self.aligner.aligned_relation_name(orig_rel_id=SYNONYMY_ID)
        if synonymy_name is None:
            raise RuntimeError(f"Cannot find mapping for synonymy id={SYNONYMY_ID}")

        syn_rels_clear = self._prepare_relations_from_list(
            rel_list=self._synonymy_as_relations(), check_embedding=True
        )
        self.logger.info(
            f"   - synonyms with both embeddings : {len(syn_rels_clear)}"
        )

        found_rels_names = set()
        for p, ch, rel_name in lu_relations_clear:
            found_rels_names.add(rel_name)
        found_rels_names.add(synonymy_name)

        self._rel_to_idx = {n: idx for idx, n in enumerate(sorted(found_rels_names))}
        self.logger.info(f"   - found different relations: {len(self._rel_to_idx)}")

        self._relations = lu_relations_clear + syn_rels_clear
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
                pbar.update(1)
                lu_emb = self.milvus_handler.get_lexical_unit_embedding(lu_id=lu.ID)
                if not lu_emb:
                    continue
                self._lu_to_emb[lu.ID] = lu_emb["embedding"]
                if limit is not None and limit > 0:
                    if len(self._lu_to_emb) >= limit:
                        break

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

    def _prepare_relations_from_list(self, rel_list, check_embedding: bool) -> List:
        relations_list = []
        for rel in rel_list:
            rel_name = self.aligner.aligned_relation_name(orig_rel_id=rel.REL_ID)
            if rel_name is None:
                self.logger.warning(
                    f"Cannot find mapping for relation {rel.REL_ID}. Skipping."
                )
                continue

            if check_embedding:
                if rel.PARENT_ID not in self._lu_to_emb:
                    continue
                if rel.CHILD_ID not in self._lu_to_emb:
                    continue

            relations_list.append([rel.PARENT_ID, rel.CHILD_ID, rel_name])
        return relations_list
