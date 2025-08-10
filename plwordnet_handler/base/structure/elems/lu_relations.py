from dataclasses import dataclass
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.general_mapper import (
    GeneralMapper,
    GeneralRelation,
)


@dataclass
class LexicalUnitRelation(GeneralRelation):
    """
    Data class representing a lexical unit relation from plWordnet database.
    """

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


class LexicalUnitRelationMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and LexicalUnitRelation objects.
    """

    map_obj = LexicalUnitRelation
