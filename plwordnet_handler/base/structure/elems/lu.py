from dataclasses import dataclass
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.comment import (
    parse_plwordnet_comment,
    ParsedComment,
)


@dataclass
class LexicalUnit:
    """
    Data class representing a lexical unit from plWordnet database.
    """

    ID: int
    lemma: str
    domain: int
    pos: int
    tagcount: int
    source: int
    status: int
    comment: ParsedComment
    variant: int
    project: int
    owner: str
    error_comment: str
    verb_aspect: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexicalUnit":
        """
        Create a LexicalUnit instance from dictionary data.

        Args:
            data: Dictionary containing lexical unit data from the database

        Returns:
            LexicalUnit: Instance of LexicalUnit dataclass

        Raises:
            KeyError: If required keys are missing from dictionary
            TypeError: If data types don't match expected types
        """
        try:
            return cls(
                ID=int(data["ID"]),
                lemma=str(data["lemma"]),
                domain=int(data["domain"]),
                pos=int(data["pos"]),
                tagcount=int(data["tagcount"]),
                source=int(data["source"]),
                status=int(data["status"]),
                comment=parse_plwordnet_comment(str(data["comment"]).strip()),
                variant=int(data["variant"]),
                project=int(data["project"]),
                owner=str(data["owner"]),
                error_comment=str(data["error_comment"]),
                verb_aspect=int(data["verb_aspect"]),
            )
        except KeyError as e:
            raise KeyError(f"Missing required key in lexical unit data: {e}")
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid data type in lexical unit data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LexicalUnit instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the lexical unit
        """
        return {
            "ID": self.ID,
            "lemma": self.lemma,
            "domain": self.domain,
            "pos": self.pos,
            "tagcount": self.tagcount,
            "source": self.source,
            "status": self.status,
            "comment": self.comment.as_dict(),
            "variant": self.variant,
            "project": self.project,
            "owner": self.owner,
            "error_comment": self.error_comment,
            "verb_aspect": self.verb_aspect,
        }

    def __str__(self) -> str:
        """
        String representation of LexicalUnit.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"LexicalUnit("
            f"ID={self.ID}, lemma='{self.lemma}', "
            f"pos={self.pos}, domain={self.domain}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of LexicalUnit.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"LexicalUnit(ID={self.ID}, lemma='{self.lemma}', domain={self.domain}, "
            f"pos={self.pos}, tagcount={self.tagcount}, source={self.source}, "
            f"status={self.status}, variant={self.variant}, project={self.project}, "
            f"owner='{self.owner}', verb_aspect={self.verb_aspect})"
        )


class LexicalUnitMapper:
    """
    Utility class for mapping between dictionary and LexicalUnit objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> LexicalUnit:
        """
        Map dictionary data to LexicalUnit object.

        Args:
            data: Dictionary containing lexical unit data

        Returns:
            LexicalUnit: Mapped LexicalUnit object
        """
        return LexicalUnit.from_dict(data)

    @staticmethod
    def map_from_dict_list(data_list: List[Dict[str, Any]]) -> List[LexicalUnit]:
        """
        Map list of dictionaries to list of LexicalUnit objects.

        Args:
            data_list: List of dictionaries containing lexical unit data

        Returns:
            List[LexicalUnit]: List of mapped LexicalUnit objects
        """
        return [LexicalUnitMapper.map_from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(lexical_unit: LexicalUnit) -> Dict[str, Any]:
        """
        Map LexicalUnit object to dictionary.

        Args:
            lexical_unit: LexicalUnit object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return lexical_unit.to_dict()

    @staticmethod
    def map_to_dict_list(lexical_units: List[LexicalUnit]) -> List[Dict[str, Any]]:
        """
        Map list of LexicalUnit objects to a list of dictionaries.

        Args:
            lexical_units: List of LexicalUnit objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [LexicalUnitMapper.map_to_dict(unit) for unit in lexical_units]
