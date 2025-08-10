from dataclasses import dataclass
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.comment import (
    parse_plwordnet_comment,
    ParsedComment,
)
from plwordnet_handler.base.structure.elems.general_mapper import (
    GeneralMapper,
    GeneralElem,
)


@dataclass
class LexicalUnit(GeneralElem):
    """
    Data class representing a lexical unit from plWordnet database.
    """

    pos: int
    lemma: str
    variant: int
    domain: int
    tagcount: int
    source: int
    project: int
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


class LexicalUnitMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and lexical units objects.
    """

    map_obj = LexicalUnit
