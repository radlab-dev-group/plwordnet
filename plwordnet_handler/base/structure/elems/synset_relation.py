from dataclasses import dataclass
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.general_mapper import (
    GeneralMapper,
    GeneralRelation,
)


@dataclass
class SynsetRelation(GeneralRelation):
    """
    Data class representing a synset relation from plWordnet database.
    """

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


class SynsetRelationMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and SynsetRelation objects.
    """

    map_obj = SynsetRelation
