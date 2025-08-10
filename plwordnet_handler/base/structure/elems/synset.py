from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from plwordnet_handler.base.structure.elems.comment import parse_plwordnet_comment
from plwordnet_handler.base.structure.elems.general_mapper import (
    GeneralMapper,
    GeneralElem,
)


@dataclass
class Synset(GeneralElem):
    """
    Data class representing a synset from plWordnet database.
    """

    split: int
    definition: str
    isabstract: int
    unitsstr: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Synset":
        """
        Create a Synset instance from dictionary data.

        Args:
            data: Dictionary containing synset data from the database

        Returns:
            Synset: Instance of Synset dataclass

        Raises:
            KeyError: If required keys are missing from dictionary
            TypeError: If data types don't match expected types
        """
        try:
            return cls(
                ID=int(data["ID"]),
                split=int(data["split"]),
                definition=str(data["definition"]),
                isabstract=int(data["isabstract"]),
                status=int(data["status"]),
                comment=parse_plwordnet_comment(str(data["comment"]).strip()),
                owner=str(data["owner"]),
                unitsstr=str(data["unitsstr"]),
                error_comment=(
                    str(data["error_comment"])
                    if data["error_comment"] is not None
                    else None
                ),
            )
        except KeyError as e:
            raise KeyError(f"Missing required key in synset data: {e}")
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid data type in synset data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Synset instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the synset
        """
        return {
            "ID": self.ID,
            "split": self.split,
            "definition": self.definition,
            "isabstract": self.isabstract,
            "status": self.status,
            "comment": self.comment.as_dict(),
            "owner": self.owner,
            "unitsstr": self.unitsstr,
            "error_comment": self.error_comment,
        }

    def __str__(self) -> str:
        """
        String representation of Synset.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"Synset("
            f"ID={self.ID}, unitsstr={self.unitsstr}, "
            f"isabstract={self.isabstract}, status={self.status}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of Synset.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"Synset(ID={self.ID}, split={self.split}, "
            f"definition='{self.definition}', "
            f"isabstract={self.isabstract}, status={self.status}, "
            f"owner='{self.owner}', unitsstr='{self.unitsstr}', "
            f"error_comment='{self.error_comment}')"
        )


class SynsetMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and Synset objects.
    """

    map_obj = Synset
