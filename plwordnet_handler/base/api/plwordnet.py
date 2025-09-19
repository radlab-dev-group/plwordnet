from tqdm import tqdm
from typing import Optional, List, Dict

import concurrent.futures

from plwordnet_handler.base.structure.elems.synset import Synset
from plwordnet_handler.base.structure.elems.lu import LexicalUnit
from plwordnet_handler.base.api.plwordnet_i import PlWordnetAPIBase
from plwordnet_handler.base.structure.elems.rel_type import RelationType
from plwordnet_handler.base.structure.elems.synset_relation import SynsetRelation
from plwordnet_handler.base.structure.elems.lu_relations import LexicalUnitRelation
from plwordnet_handler.base.structure.elems.lu_in_synset import LexicalUnitAndSynset
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.dataset.extractors.wikipedia import WikipediaExtractor


class PlWordnetAPI(PlWordnetAPIBase):
    """
    Main API class for Polish Wordnet operations.
    """

    MAX_WIKI_SENTENCES = 10

    DELEGATED_METHODS = [
        "connect",
        "disconnect",
        "is_connected",
        "get_lexical_unit",
        "get_lexical_units",
        "get_lexical_relations",
        "get_synset",
        "get_synsets",
        "get_synset_relations",
        "get_units_and_synsets",
        "get_relation_types",
    ]

    def __init__(
        self,
        connector: PlWordnetConnectorInterface,
        extract_wiki_articles: bool = False,
        use_memory_cache: bool = False,
        show_progress_bar: bool = False,
        workers_count: int = 10,
        prompts_dir: Optional[str] = None,
        prompt_name_clear_text: Optional[str] = None,
        openapi_configs_dir: Optional[str] = None,
    ):
        """
        Args:
             connector: connector interface for plWordnet
                        (PlWordnetConnectorInterface)
             extract_wiki_articles: whether to extract wiki articles
             use_memory_cache: whether to use memory caching
             show_progress_bar: whether to show tqdm progress bar
             workers_count: (int, default 10) number of workers
             used to extract wikipedia context.
             prompts_dir: str (Optional: None)
                Directory containing prompt files;
                used by PromptHandler to load the prompt.
            prompt_name_clear_text: str (Optional: None)
                The key/name of the prompt to use
                as the system prompt for correction.
            openapi_configs_dir: str (Optional: None)
                Directory containing OpenAPI config files;
        """
        super().__init__(connector)

        self.use_memory_cache = use_memory_cache
        self.show_progress_bar = show_progress_bar
        self.extract_wiki_articles = extract_wiki_articles

        self.workers_count = workers_count

        self.prompts_dir = prompts_dir
        self.prompt_name_clear_text = prompt_name_clear_text
        self.openapi_configs_dir = openapi_configs_dir

        self.corrector_handler = None
        if (
            prompts_dir is not None
            and prompt_name_clear_text is not None
            and openapi_configs_dir is not None
        ):
            self.__init_openapi_corrector_with_cache()

        self.__mem__cache_ = {}

    def get_lexical_unit(self, lu_id: int) -> Optional[LexicalUnit]:
        """
        Retrieves a lexical unit by its ID with optional memory caching.

        This method fetches a lexical unit from the data source, using an
        in-memory cache system when enabled to improve performance for repeated
        queries. The cache is organized by method name and then by lexical unit ID.

        Args:
            lu_id (int): Unique identifier of the lexical unit to retrieve

        Returns:
            Optional[LexicalUnit]: The requested lexical unit object if found,
            None if not found, or retrieval fails

        Side effects:
            - When memory cache is enabled, stores the retrieved lexical unit
              in the cache for future requests
            - Note: There appears to be a cache key inconsistency where retrieval
              uses "get_lexical_unit" but storage uses "get_lexical_unit"

        Performance:
            - First call: Fetches from a data source via connector
            - Subsequent calls: Returns a cached result when cache is enabled
        """
        if self.use_memory_cache:
            if "get_lexical_unit" in self.__mem__cache_:
                if lu_id in self.__mem__cache_["get_lexical_unit"]:
                    return self.__mem__cache_["get_lexical_unit"][lu_id]

        lu = self.connector.get_lexical_unit(lu_id=lu_id)
        if self.use_memory_cache:
            if "get_lexical_unit" not in self.__mem__cache_:
                self.__mem__cache_["get_lexical_unit"] = {}
            self.__mem__cache_["get_lexical_unit"][lu_id] = lu
        return lu

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units from the wordnet connector. Additional memory
        caching for better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical units or None if error
        """
        if self.use_memory_cache:
            if "get_lexical_units" in self.__mem__cache_:
                return self.__mem__cache_["get_lexical_units"]

        lu_list = self.connector.get_lexical_units(limit=limit)
        if self.extract_wiki_articles:
            if self.workers_count > 1:
                lu_list = self.__add_wiki_context_parallel(
                    lu_list=lu_list,
                    force_download_content=True,
                    workers_count=self.workers_count,
                )
            else:
                lu_list = self.__add_wiki_context(
                    lu_list=lu_list,
                    force_download_content=True,
                )

        # extractor = WikipediaExtractor(
        #     max_sentences=self.MAX_WIKI_SENTENCES,
        #     prompts_dir=self.prompts_dir,
        #     clear_text_prompt_name=self.prompt_name_clear_text,
        #     openapi_configs_dir=self.openapi_configs_dir,
        # )

        if self.use_memory_cache:
            self.__mem__cache_["get_lexical_units"] = lu_list

        return lu_list

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations from the wordnet connector. Additional memory
        caching for better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if an error occurred
        """
        if self.use_memory_cache:
            if "get_lexical_relations" in self.__mem__cache_:
                return self.__mem__cache_["get_lexical_relations"]

        lu_rels = self.connector.get_lexical_relations(limit=limit)
        if self.use_memory_cache:
            self.__mem__cache_["get_lexical_relations"] = lu_rels

        return lu_rels

    def get_synset(self, syn_id: int) -> Optional[Synset]:
        """
        Retrieves a synset by its ID with optional memory caching.

        This method fetches a synset from the data source, using an in-memory
        cache system when enabled to improve performance for repeated queries.
        The cache is organized by method name and then by synset ID.

        Args:
            syn_id (int): Unique identifier of the synset to retrieve

        Returns:
            Optional[Synset]: The requested synset object if found,
            None if not found, or retrieval fails

        Side effects:
            - When memory cache is enabled, stores the retrieved synset
              in the cache for future requests
            - Initializes the cache structure if it doesn't exist

        Performance:
            - First call: Fetches from a data source via connector
            - Subsequent calls: Returns a cached result when cache is enabled
        """
        if self.use_memory_cache:
            if "get_synset" in self.__mem__cache_:
                if syn_id in self.__mem__cache_["get_synset"]:
                    return self.__mem__cache_["get_synset"][syn_id]

        synset = self.connector.get_synset(syn_id=syn_id)
        if self.use_memory_cache:
            if "get_synset" not in self.__mem__cache_:
                self.__mem__cache_["get_synset"] = {}
            self.__mem__cache_["get_synset"][syn_id] = synset
        return synset

    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Get synset from the wordnet connector. Additional memory caching
        for better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of Synset or None if an error occurred
        """
        if self.use_memory_cache:
            if "get_synsets" in self.__mem__cache_:
                return self.__mem__cache_["get_synsets"]

        syn_list = self.connector.get_synsets(limit=limit)
        if self.use_memory_cache:
            self.__mem__cache_["get_synsets"] = syn_list

        return syn_list

    def get_synset_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[SynsetRelation]]:
        """
        Get synset relations from the wordnet connector.
        Additional memory caching to better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of SynsetRelation or None if an error occurred
        """
        if self.use_memory_cache:
            if "get_synset_relations" in self.__mem__cache_:
                return self.__mem__cache_["get_synset_relations"]

        syn_rels = self.connector.get_synset_relations(limit=limit)
        if self.use_memory_cache:
            self.__mem__cache_["get_synset_relations"] = syn_rels

        return syn_rels

    def get_units_and_synsets(
        self, limit: Optional[int] = None, return_mapping: bool = False
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Get units in synset from the wordnet connector.
        Additional memory caching to better performance is available.

        Args:
            limit: Optional limit for the number of results
            return_mapping: Returns a mapping of synset to lexical units

        Returns:
            List of LexicalUnitAndSynset or None if an error occurred
        """
        if self.use_memory_cache:
            if (
                "get_units_and_synsets" in self.__mem__cache_
                and return_mapping in self.__mem__cache_["get_units_and_synsets"]
            ):
                return self.__mem__cache_["get_units_and_synsets"][return_mapping]

        u_a_s = self.connector.get_units_and_synsets(limit=limit)
        if return_mapping:
            u_a_s = self._map_units_to_synsets(uas=u_a_s)

        if self.use_memory_cache:
            if "get_units_and_synsets" not in self.__mem__cache_:
                self.__mem__cache_["get_units_and_synsets"] = {}
            if return_mapping not in self.__mem__cache_["get_units_and_synsets"]:
                self.__mem__cache_["get_units_and_synsets"][return_mapping] = u_a_s
        return u_a_s

    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Get relation types from the wordnet connector.
        Additional memory caching to better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of relation types or None if an error occurred
        """
        if self.use_memory_cache:
            if "get_relation_types" in self.__mem__cache_:
                return self.__mem__cache_["get_relation_types"]

        rel_types = self.connector.get_relation_types(limit=limit)
        if self.use_memory_cache:
            self.__mem__cache_["get_relation_types"] = rel_types

        return rel_types

    def __init_openapi_corrector_with_cache(self):
        """
        Initialize an OpenAPI handler with caching
        for Wikipedia description extraction.

        Creates an :class:`OpenApiHandlerWithCache` instance configured
        to load prompts from `prompts_dir` using `prompt_name` and to store
        cached API responses in a dedicated work directory. `openapi_configs_dir`
        points to the directory containing OpenAPI specification files
        required by the handler.

        The handler instance is stored in `self.api_handler` for later use.
        """
        from rdl_ml_utils.open_api.cache_api import OpenApiHandlerWithCache

        self.corrector_handler = OpenApiHandlerWithCache(
            prompts_dir=self.prompts_dir,
            prompt_name=self.prompt_name_clear_text,
            workdir="./__cache/wikipedia_description/",
            openapi_configs_dir=self.openapi_configs_dir,
            max_workers=None,
        )

    def _fix_content(self, content_str: str) -> str:
        assert (
            self.corrector_handler is not None
        ), "Corrector handler not initialized!"

        if not len(content_str):
            return ""
        return self.corrector_handler.generate(text_str=content_str)

    def __add_wiki_context(
        self,
        lu_list: List[LexicalUnit],
        force_download_content: bool = False,
    ):
        """
        Enriches lexical units with Wikipedia content descriptions.

        This private method processes a list of lexical units and adds Wikipedia
        content to their external URL descriptions when available. It extracts
        the main description from Wikipedia pages linked to each lexical unit.

        Args:
            lu_list (List[LexicalUnit]): List of lexical units
            to enrich with Wiki content
            force_download_content (bool): If True, downloads content even
            if it already exists; if False, skips units that already have content

        Returns:
            List[LexicalUnit]: The same list of lexical units with
            added Wikipedia content

        Side effects:
            - Modifies the content field of external_url_description
            for each lexical unit
            - Shows a progress bar if show_progress_bar is enabled
            - Logs processing information for each lexical unit
            when a progress bar is disabled

        Processing logic:
            - Skips lexical units without external URL descriptions
            - Skips lexical units with empty or whitespace-only URLs
            - Skips lexical units that already have content
            (unless force_download_content is True)
            - Uses WikipediaExtractor to fetch content with a sentence limit
        """
        pbar = None
        if self.show_progress_bar:
            pbar = tqdm(total=len(lu_list), desc="Adding Wiki context")

        extractor = WikipediaExtractor(max_sentences=self.MAX_WIKI_SENTENCES)

        for lu in lu_list:
            if pbar:
                pbar.update(1)
            else:
                self.logger.info(
                    f"str(lu) -> has url {lu.comment.external_url_description}"
                )

            if not lu.comment.external_url_description:
                continue

            url = lu.comment.external_url_description.url
            if not url or not len(url.strip()):
                continue

            content = lu.comment.external_url_description.content
            if content and content.strip():
                if not force_download_content:
                    continue

            content = extractor.extract_main_description(wikipedia_url=url)
            if not content:
                continue
            lu.comment.external_url_description.content = content
        return lu_list

    def __add_wiki_context_parallel(
        self,
        lu_list: List[LexicalUnit],
        workers_count: int = 4,
        force_download_content: bool = False,
    ) -> List[LexicalUnit]:
        """
        Enriches lexical units with Wikipedia content in parallel using threads.

        The list of lexical units is split into `workers_count` chunks and each
        chunk is processed by `__add_wiki_context` in a separate thread.

        Args:
            lu_list: List of lexical units to enrich.
            workers_count: Number of worker threads to spawn.
            force_download_content: Passed to `__add_wiki_context`.

        Returns:
            List[LexicalUnit]: The combined list with Wikipedia content is added.
        """
        if not lu_list:
            return []

        workers = max(1, min(workers_count, len(lu_list)))

        # Split the list into roughly equal chunks
        chunk_size = (len(lu_list) + workers - 1) // workers
        chunks = [
            lu_list[i : i + chunk_size] for i in range(0, len(lu_list), chunk_size)
        ]

        results: List[LexicalUnit] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit each chunk to the executor
            futures = [
                executor.submit(
                    self.__add_wiki_context,
                    chunk,
                    force_download_content,
                )
                for chunk in chunks
            ]

            # Gather results while preserving original order
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                if chunk_result:
                    results.extend(chunk_result)
        return results

    @staticmethod
    def _map_units_to_synsets(
        uas: List[LexicalUnitAndSynset],
    ) -> Dict[int, List[int]]:
        """
        Create a mapping from the lexical unit and synset,
        to collections of lexical unit IDs.

        Args:
            uas: List of LexicalUnitAndSynset objects containing
            relationships between lexical units and synsets

        Returns:
            Dict[int, List[int]]: Dictionary mapping synset IDs to sets of
            associated lexical unit IDs
        """

        _map = {}
        for u in uas:
            if u.SYN_ID not in _map:
                _map[u.SYN_ID] = set()
            _map[u.SYN_ID].add(u.LEX_ID)
        return _map
