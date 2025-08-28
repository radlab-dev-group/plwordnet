import os
import json
from typing import List, Optional, Dict

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
        """
        Initialize a mapper that aligns original WordNet identifiers with a
        contiguous 0‑based index space required by the RelGAT dataset.

        The constructor works in two exclusive modes:

        * **Mapping‑creation mode** (`prepare_mapping=True`):
          - ``plwn_api`` must be supplied; the mapper will query the API for
            lexical‑unit and relation‑type information and build fresh mapping
            tables.
          - No ``mapping_path`` is needed.

        * **Mapping‑loading mode** (`prepare_mapping=False`):
          - ``mapping_path`` must point to a directory that contains the four JSON
            files produced by a previous run:
            ``rel_align_to_original.json``, ``rel_original_to_align.json``,
            ``rel_name_to_aligned_id.json`` and ``aligned_id_to_rel_name.json``.
          - The mapper simply reads those files into memory.

        Parameters
        ----------
        plwn_api : PolishWordnet, optional
            An instantiated PolishWordnet API client. Required when
            ``prepare_mapping=True``.
        prepare_mapping : bool, default ``False``
            Switch between “create new mappings” and “load existing mappings.”
        mapping_path : str, optional
            Directory that holds the JSON mapping files. Required when
            ``prepare_mapping=False``.
        log_level : str, default ``"INFO"``
            Logging verbosity for the internal logger.
        logger_file_name : str, optionally
            If supplied, a file handler will be added to the logger and logs will be
            written to this file.

        Raises
        ------
        Exception
            * If ``prepare_mapping=True`` and ``plwn_api`` is ``None``.
            * If ``prepare_mapping=False`` and ``mapping_path`` is ``None``.
            * If any required mapping file is missing or cannot be parsed
              during loading.

        Notes
        -----
        The method does **not** perform any heavy computation; it only prepares
        internal data structures and the logger.  Actual mapping generation is
        delegated to: `_prepare_mapping`.
        """
        self.plwn_api = plwn_api
        self.mapping_path = mapping_path
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

    @property
    def rel_name_to_aligned_idx_dict(self) -> Dict[str, int]:
        """
        Mapping from the **human‑readable relation name** (e.g. ``"hypernym"``)
        to the **aligned integer identifier** used by the RelGAT dataset.

        Returns
        -------
        dict[str, int]
            A dictionary where keys are relation names and values are the
            corresponding 0‑based aligned IDs.

        Notes
        -----
        The dictionary is populated either by `_prepare_mapping` or by
        `_load_mappings_from_dir`.  It is safe to read but must not be
        modified directly – treat it as an immutable view.
        """
        return self._rel_name_to_aligned_id

    def aligned_relation_id(self, orig_rel_id: int) -> Optional[int]:
        """
        Translate an **original WordNet relation ID** (as stored in the
        database) to the **aligned 0‑based index**.

        Parameters
        ----------
        orig_rel_id : int
            The original relation identifier (the ``ID`` attribute of RelationType).

        Returns
        -------
        int or None
            The aligned identifier if the original ID is known, otherwise ``None``.

        Notes
        -----
        The lookup uses the reverse mapping ``_rel_original_to_align`` that is
        built during mapping creation or loading.
        """
        return self._rel_original_to_align.get(str(orig_rel_id), None)

    def original_relation_name(self, aligned_rel_id: Optional[int]) -> Optional[str]:
        """
        Retrieve the **original WordNet relation name** from an aligned identifier.

        Parameters
        ----------
        aligned_rel_id : int or None
            The aligned 0‑based identifier produced by the mapper.

        Returns
        -------
        str or None
            The original relation name (e.g. ``"hypernym"``) if the aligned ID is
            present in the mapping; otherwise ``None``.

        Raises
        ------
        None – the method simply returns ``None`` for unknown IDs.

        Notes
        -----
        This is the inverse operation of `aligned_relation_id` but works on
        the *name* rather than the numeric ID.
        """
        if aligned_rel_id is None:
            return None
        return self._aligned_id_to_rel_name.get(aligned_rel_id, None)

    def aligned_relation_name(self, orig_rel_id: Optional[int]) -> Optional[str]:
        """
        Resolve the **aligned (0‑based) identifier** for a given original
        WordNet relation ID and then return its **human‑readable name**.

        This convenience wrapper combines `aligned_relation_id` and the
        internal ``_aligned_id_to_rel_name`` dictionary.

        Parameters
        ----------
        orig_rel_id : int or None
            The original relation identifier from WordNet.

        Returns
        -------
        str or None
            The relation name if a mapping exists; ``None`` otherwise.

        Notes
        -----
        * ``orig_rel_id`` may be ``None`` – the method will simply return ``None``.
        * The lookup is performed with string keys because the mapping tables
          store IDs as strings when they are read from JSON.
        """
        aligned_rel_id = (
            self.aligned_relation_id(orig_rel_id=orig_rel_id)
            if orig_rel_id
            else None
        )
        if aligned_rel_id is None:
            return None

        a_rel_name = self._aligned_id_to_rel_name.get(str(aligned_rel_id), None)
        return a_rel_name

    def export_to_dir(self, out_directory: str) -> None:
        """
        Serialize all four mapping tables to JSON files inside ``out_directory``.

        The files written are:

        * ``rel_align_to_original.json`` – aligned ID → original WordNet ID
        * ``rel_original_to_align.json`` – original ID → aligned ID
        * ``rel_name_to_aligned_id.json`` – relation name → aligned ID
        * ``aligned_id_to_rel_name.json`` – aligned ID → relation name

        Parameters
        ----------
        out_directory : str
            Destination directory. It will be created if it does not already exist.

        Raises
        ------
        Exception
            Propagates any I/O error that occurs while opening or writing the
            files.

        Notes
        -----
        The method writes the mappings in a pretty‑printed JSON format
        (indentation=2, ``ensure_ascii=False``) to keep the files human‑readable.
        """
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
        """
        Build fresh mapping tables from the PolishWordnet API.

        The method delegates the heavy lifting to `_prepare_relations`,
        which in turn calls the private ``__align_rel_identifiers`` helper.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error raised while querying the API for relation
            types or while aligning identifiers.

        Notes
        -----
        This private method is invoked automatically when the class is
        instantiated with ``prepare_mapping=True``.
        """
        self.logger.info("Preparing relations and lexical units mapping")
        self._prepare_relations()

    def _load_mappings_from_dir(self, mapping_path: str) -> None:
        """
        Populate the internal mapping dictionaries by reading the JSON files
        stored in ``mapping_path``.

        Parameters
        ----------
        mapping_path : str
            Path to a directory that must contain the four JSON files
            expected by the aligner.

        Raises
        ------
        Exception
            * If any of the required files is missing.
            * If a file cannot be parsed as valid JSON.
            * If a file has an unexpected extension (anything other than ``.json``).

        Notes
        -----
        After successful loading the four public attributes
        ``_rel_align_to_original``, ``_rel_original_to_align``,
        ``_rel_name_to_aligned_id`` and ``_aligned_id_to_rel_name`` are
        updated in‑place so that later calls operate on the loaded data.
        """
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

        self._rel_align_to_original = self._filename_to_mapping[
            "rel_align_to_original.json"
        ]
        self._rel_original_to_align = self._filename_to_mapping[
            "rel_original_to_align.json"
        ]
        self._rel_name_to_aligned_id = self._filename_to_mapping[
            "rel_name_to_aligned_id.json"
        ]
        self._aligned_id_to_rel_name = self._filename_to_mapping[
            "aligned_id_to_rel_name.json"
        ]

    def _prepare_relations(self):
        """
        Trigger the alignment of relation‑type identifiers.

        This private helper gets the list of all relation types from the
        ``PolishWordnet`` instance (via ``self.plwn_api.get_relation_types()``)
        and forwards it to `__align_rel_identifiers`.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error raised while fetching relation types from the API.
        """
        self.logger.info(" - preparing relations mappings")
        self.__align_rel_identifiers(rels_list=self.plwn_api.get_relation_types())

    def __align_rel_identifiers(self, rels_list: List[RelationType]) -> None:
        """
        Create the four bi‑directional mapping tables that translate between the
        original WordNet relation identifiers and the contiguous 0‑based IDs used
        by the RelGAT dataset.

        For every ``RelationType`` object in ``rels_list`` the method:

        1. Assigns a new aligned ID (the enumeration index, starting at 0).
        2. Stores the original numeric ``ID`` under that aligned ID.
        3. Records the reverse mapping (original → aligned).
        4. Associates the textual ``name`` with the aligned ID.
        5. Stores the inverse (aligned ID → name).

        Parameters
        ----------
        rels_list : list[RelationType]
            A list of relation‑type objects obtained from the WordNet API.
            Each element must expose the attributes ``ID`` (original numeric ID)
            and ``name`` (human‑readable label).

        Returns
        -------
        None

        Raises
        ------
        None – the method never raises; missing or malformed entries are simply
        ignored (the internal dictionaries are cleared before filling).

        Note
        -----
        * The resulting dictionaries are:
            * ``_rel_align_to_original`` – aligned ID → original numeric ID
            * ``_rel_original_to_align`` – original numeric ID → aligned ID
            * ``_rel_name_to_aligned_id`` – relation name → aligned ID
            * ``_aligned_id_to_rel_name`` – aligned ID → relation name
        * All four structures are cleared at the start of the method so that
        a fresh alignment can be performed repeatedly without residual data.
        """
        self.logger.info(
            f"   - number of relations to align identifiers {len(rels_list)}"
        )

        self._rel_align_to_original.clear()
        self._rel_original_to_align.clear()
        self._rel_name_to_aligned_id.clear()
        self._aligned_id_to_rel_name.clear()
        for idx, rel in enumerate(rels_list):
            self._rel_align_to_original[idx] = int(rel.ID)
            self._rel_original_to_align[int(rel.ID)] = idx
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
