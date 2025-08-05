from tqdm import tqdm
from typing import Optional, List

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
        "get_lexical_units",
        "get_lexical_relations",
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
    ):
        """
        Args:
             connector: connector interface for plWordnet
                        (PlWordnetConnectorInterface)
             extract_wiki_articles: whether to extract wiki articles
             use_memory_cache: whether to use memory caching
             show_progress_bar: whether to show tqdm progress bar
        """
        super().__init__(connector)

        self.use_memory_cache = use_memory_cache
        self.show_progress_bar = show_progress_bar
        self.extract_wiki_articles = extract_wiki_articles

        self.__mem__cache_ = {}

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units from the wordnet connector.
        Additional memory caching to better performance is available.

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
            lu_list = self.__add_wiki_context(
                lu_list=lu_list, force_download_content=True
            )

        if self.use_memory_cache:
            self.__mem__cache_["get_lexical_units"] = lu_list

        return lu_list

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations from the wordnet connector.
        Additional memory caching to better performance is available.

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

    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Get synset from the wordnet connector.
        Additional memory caching to better performance is available.

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
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Get units in synset from the wordnet connector.
        Additional memory caching to better performance is available.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitAndSynset or None if an error occurred
        """
        if self.use_memory_cache:
            if "get_units_and_synsets" in self.__mem__cache_:
                return self.__mem__cache_["get_units_and_synsets"]

        u_a_s = self.connector.get_units_and_synsets(limit=limit)
        if self.use_memory_cache:
            self.__mem__cache_["get_units_and_synsets"] = u_a_s

        return u_a_s

    def __add_wiki_context(
        self, lu_list: List[LexicalUnit], force_download_content: bool = False
    ):
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
