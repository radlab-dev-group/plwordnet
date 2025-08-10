from dataclasses import dataclass
from typing import Dict, Any, List


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


class LexicalUnitAndSynsetMapper:
    """
    Utility class for mapping between dictionary and LexicalUnitAndSynset objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> LexicalUnitAndSynset:
        """
        Map dictionary data to LexicalUnitAndSynset object.

        Args:
            data: Dictionary containing lexical unit and synset relationship data

        Returns:
            LexicalUnitAndSynset: Mapped LexicalUnitAndSynset object
        """
        return LexicalUnitAndSynset.from_dict(data)

    @staticmethod
    def map_from_dict_list(
        data_list: List[Dict[str, Any]]
    ) -> List[LexicalUnitAndSynset]:
        """
        Map list of dictionaries to list of LexicalUnitAndSynset objects.

        Args:
            data_list: List of dictionaries containing
            lexical unit and synset relationship data

        Returns:
            List[LexicalUnitAndSynset]: List of mapped LexicalUnitAndSynset objects
        """
        return [LexicalUnitAndSynset.from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(lu_and_synset: LexicalUnitAndSynset) -> Dict[str, Any]:
        """
        Map LexicalUnitAndSynset object to dictionary.

        Args:
            lu_and_synset: LexicalUnitAndSynset object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return lu_and_synset.to_dict()

    @staticmethod
    def map_to_dict_list(
        lu_and_synsets: List[LexicalUnitAndSynset],
    ) -> List[Dict[str, Any]]:
        """
        Map list of LexicalUnitAndSynset objects to list of dictionaries.

        Args:
            lu_and_synsets: List of LexicalUnitAndSynset objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [lu_synset.to_dict() for lu_synset in lu_and_synsets]
