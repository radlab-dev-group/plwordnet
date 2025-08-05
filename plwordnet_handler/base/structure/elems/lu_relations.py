from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class LexicalUnitRelation:
    """
    Data class representing a lexical unit relation from plWordnet database.
    """

    PARENT_ID: int
    CHILD_ID: int
    REL_ID: int
    valid: int
    owner: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexicalUnitRelation":
        """
        Create a LexicalUnitRelation instance from dictionary data.

        Args:
            data: Dictionary containing lexical unit relation data from the database

        Returns:
            LexicalUnitRelation: Instance of LexicalUnitRelation dataclass

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
            raise KeyError(
                f"Missing required key in lexical unit relation data: {e}"
            )
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid data type in lexical unit relation data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LexicalUnitRelation instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the lexical unit relation
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
        String representation of LexicalUnitRelation.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"LexicalUnitRelation("
            f"PARENT_ID={self.PARENT_ID}, CHILD_ID={self.CHILD_ID}, "
            f"REL_ID={self.REL_ID}, valid={self.valid}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of LexicalUnitRelation.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"LexicalUnitRelation("
            f"PARENT_ID={self.PARENT_ID}, CHILD_ID={self.CHILD_ID}, "
            f"REL_ID={self.REL_ID}, "
            f"valid={self.valid}, owner='{self.owner}')"
        )

    @property
    def is_valid(self) -> bool:
        """
        Check if the lexical unit relation has valid basic properties.

        Returns:
            bool: True if a lexical unit relation appears valid, False otherwise
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
        Check if the lexical unit relation is active (valid=1).

        Returns:
            bool: True if the relation is active, False otherwise
        """
        return self.valid == 1

    @property
    def has_owner(self) -> bool:
        """
        Check if the lexical unit relation has an owner.

        Returns:
            bool: True if an owner exists and is not empty, False otherwise
        """
        return bool(self.owner.strip())


class LexicalUnitRelationMapper:
    """
    Utility class for mapping between dictionary and LexicalUnitRelation objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> LexicalUnitRelation:
        """
        Map dictionary data to a LexicalUnitRelation object.

        Args:
            data: Dictionary containing lexical unit relation data

        Returns:
            LexicalUnitRelation: Mapped LexicalUnitRelation object
        """
        return LexicalUnitRelation.from_dict(data)

    @staticmethod
    def map_from_dict_list(
        data_list: List[Dict[str, Any]]
    ) -> List[LexicalUnitRelation]:
        """
        Map a list of dictionaries to a list of LexicalUnitRelation objects.

        Args:
            data_list: List of dictionaries containing lexical unit relation data

        Returns:
            List[LexicalUnitRelation]: List of mapped LexicalUnitRelation objects
        """
        return [LexicalUnitRelation.from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(lexical_unit_relation: LexicalUnitRelation) -> Dict[str, Any]:
        """
        Map LexicalUnitRelation object to dictionary.

        Args:
            lexical_unit_relation: LexicalUnitRelation object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return lexical_unit_relation.to_dict()

    @staticmethod
    def map_to_dict_list(
        lexical_unit_relations: List[LexicalUnitRelation],
    ) -> List[Dict[str, Any]]:
        """
        Map list of LexicalUnitRelation objects to a list of dictionaries.

        Args:
            lexical_unit_relations: List of LexicalUnitRelation objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [relation.to_dict() for relation in lexical_unit_relations]
