from abc import ABC, abstractmethod
from typing import Optional, List

from plwordnet_handler.base.structure.elems.synset import Synset
from plwordnet_handler.base.structure.elems.lu import LexicalUnit
from plwordnet_handler.base.structure.elems.rel_type import RelationType
from plwordnet_handler.base.structure.elems.synset_relation import SynsetRelation
from plwordnet_handler.base.structure.elems.lu_in_synset import LexicalUnitAndSynset
from plwordnet_handler.base.structure.elems.lu_relations import LexicalUnitRelation


class PlWordnetConnectorInterface(ABC):
    """
    Abstract interface for plWordnet database connectors.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection using connector.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection from connector.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if a connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        pass

    @abstractmethod
    def get_lexical_unit(self, lu_id: int) -> Optional[LexicalUnit]:
        """
        Get a lexical unit with a given `lu_id`

        Args:
            lu_id: Lexical unit id to find

        Returns:
            LexicalUnit or None if lu_id is not found
        """
        pass

    @abstractmethod
    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical units or None if error
        """
        pass

    @abstractmethod
    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if an error occurs
        """
        pass

    @abstractmethod
    def get_synset(self, syn_id: int) -> Optional[Synset]:
        """
        Get synset with given synset id

        Args:
            syn_id: Synset id

        Returns:
            Synset or None if syn_id is not found
        """
        pass

    @abstractmethod
    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Get synsets

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of Synsets or None if error
        """
        pass

    @abstractmethod
    def get_synset_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[SynsetRelation]]:
        """
        Get synset relations

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of SynsetRelation or None if error
        """
        pass

    @abstractmethod
    def get_units_and_synsets(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Get units and synset

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitAndSynset or None if error
        """
        pass

    @abstractmethod
    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Get types of relations

        Args:
            limit: Optional limit for a number of results

        Returns:
            List of relation types or None if error occurred
        """
        pass

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        if self.connect():
            return self
        else:
            raise ConnectionError("Failed to establish database connection")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.disconnect()
