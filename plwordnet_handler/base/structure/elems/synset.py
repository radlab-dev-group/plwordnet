from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from plwordnet_handler.base.structure.elems.comment import (
    parse_plwordnet_comment,
    ParsedComment,
)


@dataclass
class Synset:
    """
    Data class representing a synset from plWordnet database.
    """

    ID: int
    split: int
    definition: str
    isabstract: int
    status: int
    comment: ParsedComment
    owner: str
    unitsstr: str
    error_comment: Optional[str]

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


class SynsetMapper:
    """
    Utility class for mapping between dictionary and Synset objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> Synset:
        """
        Map dictionary data to a Synset object.

        Args:
            data: Dictionary containing synset data

        Returns:
            Synset: Mapped Synset object
        """
        return Synset.from_dict(data)

    @staticmethod
    def map_from_dict_list(data_list: List[Dict[str, Any]]) -> List[Synset]:
        """
        Map list of dictionaries to list of Synset objects.

        Args:
            data_list: List of dictionaries containing synset data

        Returns:
            List[Synset]: List of mapped Synset objects
        """
        return [SynsetMapper.map_from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(synset: Synset) -> Dict[str, Any]:
        """
        Map Synset object to dictionary.

        Args:
            synset: Synset object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return synset.to_dict()

    @staticmethod
    def map_to_dict_list(synsets: List[Synset]) -> List[Dict[str, Any]]:
        """
        Map list of Synset objects to list of dictionaries.

        Args:
            synsets: List of Synset objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [SynsetMapper.map_to_dict(synset) for synset in synsets]
