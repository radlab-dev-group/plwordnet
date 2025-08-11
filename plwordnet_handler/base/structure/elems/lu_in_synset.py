from dataclasses import dataclass
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.general_mapper import GeneralMapper


@dataclass
class LexicalUnitAndSynsetFakeRelation:
    """
    Data class representing a fake relation between lexical units.

    This class simulates a relation structure for lexical units associations
    by providing the standard relation attributes (parent, child, relation type).

    Attributes:
        PARENT_ID: Identifier of the parent element in the relation
        CHILD_ID: Identifier of the child element in the relation
        REL_ID: Identifier of the relation type
    """

    PARENT_ID: int
    CHILD_ID: int
    REL_ID: int


@dataclass
class LexicalUnitAndSynset:
    """
    Data class representing a lexical unit
    and synset relationship from plWordnet database.
    """

    LEX_ID: int
    SYN_ID: int
    unitindex: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexicalUnitAndSynset":
        """
        Create a LexicalUnitAndSynset instance from dictionary data.

        Args:
            data: Dictionary containing lexical unit and
            synset relationship data from the database

        Returns:
            LexicalUnitAndSynset: Instance of LexicalUnitAndSynset dataclass

        Raises:
            KeyError: If required keys are missing from dictionary
            TypeError: If data types don't match expected types
        """
        try:
            return cls(
                LEX_ID=int(data["LEX_ID"]),
                SYN_ID=int(data["SYN_ID"]),
                unitindex=int(data["unitindex"]),
            )
        except KeyError as e:
            raise KeyError(
                f"Missing required key in lexical unit and synset data: {e}"
            )
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Invalid data type in lexical unit and synset data: {e}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LexicalUnitAndSynset instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
            of the lexical unit and synset relationship
        """
        return {
            "LEX_ID": self.LEX_ID,
            "SYN_ID": self.SYN_ID,
            "unitindex": self.unitindex,
        }

    def __str__(self) -> str:
        """
        String representation of LexicalUnitAndSynset.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"LexicalUnitAndSynset("
            f"LEX_ID={self.LEX_ID}, SYN_ID={self.SYN_ID}, "
            f"unitindex={self.unitindex}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of LexicalUnitAndSynset.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"LexicalUnitAndSynset(LEX_ID={self.LEX_ID}, SYN_ID={self.SYN_ID}, "
            f"unitindex={self.unitindex})"
        )


class LexicalUnitAndSynsetMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and LexicalUnitAndSynset objects.
    """

    map_obj = LexicalUnitAndSynset
