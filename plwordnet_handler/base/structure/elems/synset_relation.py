from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class SynsetRelation:
    """
    Data class representing a synset relation from plWordnet database.
    """

    PARENT_ID: int
    CHILD_ID: int
    REL_ID: int
    valid: int
    owner: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynsetRelation":
        """
        Create a SynsetRelation instance from dictionary data.

        Args:
            data: Dictionary containing synset relation data from the database

        Returns:
            SynsetRelation: Instance of SynsetRelation dataclass

        Raises:
            KeyError: If required keys are missing from dictionary
            TypeError: If data types don't match expected types
        """
        try:
            return cls(
                PARENT_ID=int(data["PARENT_ID"]),
                CHILD_ID=int(data["CHILD_ID"]),
                REL_ID=int(data["REL_ID"]),
                valid=int(data["valid"]),
                owner=str(data["owner"]),
            )
        except KeyError as e:
            raise KeyError(f"Missing required key in synset relation data: {e}")
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid data type in synset relation data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SynsetRelation instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the synset relation
        """
        return {
            "PARENT_ID": self.PARENT_ID,
            "CHILD_ID": self.CHILD_ID,
            "REL_ID": self.REL_ID,
            "valid": self.valid,
            "owner": self.owner,
        }

    def __str__(self) -> str:
        """
        String representation of SynsetRelation.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"SynsetRelation("
            f"PARENT_ID={self.PARENT_ID}, CHILD_ID={self.CHILD_ID}, "
            f"REL_ID={self.REL_ID}, valid={self.valid}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of SynsetRelation.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"SynsetRelation(PARENT_ID={self.PARENT_ID}, CHILD_ID={self.CHILD_ID}, "
            f"REL_ID={self.REL_ID}, valid={self.valid}, owner='{self.owner}')"
        )

    @property
    def is_valid(self) -> bool:
        """
        Check if the synset relation has valid basic properties.

        Returns:
            bool: True if a synset relation appears valid, False otherwise
        """
        return (
            self.PARENT_ID > 0
            and self.CHILD_ID > 0
            and self.REL_ID > 0
            and self.valid in [0, 1]
        )

    @property
    def is_active(self) -> bool:
        """
        Check if the synset relation is active (valid=1).

        Returns:
            bool: True if the relation is active, False otherwise
        """
        return self.valid == 1

    @property
    def has_owner(self) -> bool:
        """
        Check if the synset relation has an owner.

        Returns:
            bool: True if an owner exists and is not empty, False otherwise
        """
        return bool(self.owner.strip())


class SynsetRelationMapper:
    """
    Utility class for mapping between dictionary and SynsetRelation objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> SynsetRelation:
        """
        Map dictionary data to SynsetRelation object.

        Args:
            data: Dictionary containing synset relation data

        Returns:
            SynsetRelation: Mapped SynsetRelation object
        """
        return SynsetRelation.from_dict(data)

    @staticmethod
    def map_from_dict_list(data_list: List[Dict[str, Any]]) -> List[SynsetRelation]:
        """
        Map list of dictionaries to list of SynsetRelation objects.

        Args:
            data_list: List of dictionaries containing synset relation data

        Returns:
            List[SynsetRelation]: List of mapped SynsetRelation objects
        """
        return [SynsetRelation.from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(synset_relation: SynsetRelation) -> Dict[str, Any]:
        """
        Map SynsetRelation object to dictionary.

        Args:
            synset_relation: SynsetRelation object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return synset_relation.to_dict()

    @staticmethod
    def map_to_dict_list(
        synset_relations: List[SynsetRelation],
    ) -> List[Dict[str, Any]]:
        """
        Map list of SynsetRelation objects to list of dictionaries.

        Args:
            synset_relations: List of SynsetRelation objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [relation.to_dict() for relation in synset_relations]
