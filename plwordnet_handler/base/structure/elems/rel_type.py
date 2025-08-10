from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from plwordnet_handler.base.structure.elems.general_mapper import GeneralMapper


@dataclass
class RelationType:
    """
    Data class representing a relation type from plWordnet database.
    """

    ID: int
    objecttype: int
    PARENT_ID: Optional[int]
    REVERSE_ID: Optional[int]
    name: str
    description: str
    posstr: str
    autoreverse: int
    display: str
    shortcut: str
    pwn: str
    order: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationType":
        """
        Create a RelationType instance from dictionary data.

        Args:
            data: Dictionary containing relation type data from the database

        Returns:
            RelationType: Instance of RelationType dataclass

        Raises:
            KeyError: If required keys are missing from dictionary
            TypeError: If data types don't match expected types
        """
        try:
            return cls(
                ID=int(data["ID"]),
                objecttype=int(data["objecttype"]),
                PARENT_ID=(
                    int(data["PARENT_ID"]) if data["PARENT_ID"] is not None else None
                ),
                REVERSE_ID=(
                    int(data["REVERSE_ID"])
                    if data["REVERSE_ID"] is not None
                    else None
                ),
                name=str(data["name"]),
                description=str(data["description"]),
                posstr=str(data["posstr"]),
                autoreverse=int(data["autoreverse"]),
                display=str(data["display"]),
                shortcut=str(data["shortcut"]),
                pwn=str(data["pwn"]),
                order=int(data["order"]),
            )
        except KeyError as e:
            raise KeyError(f"Missing required key in relation type data: {e}")
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid data type in relation type data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RelationType instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the relation type
        """
        return {
            "ID": self.ID,
            "objecttype": self.objecttype,
            "PARENT_ID": self.PARENT_ID,
            "REVERSE_ID": self.REVERSE_ID,
            "name": self.name,
            "description": self.description,
            "posstr": self.posstr,
            "autoreverse": self.autoreverse,
            "display": self.display,
            "shortcut": self.shortcut,
            "pwn": self.pwn,
            "order": self.order,
        }

    def __str__(self) -> str:
        """
        String representation of RelationType.

        Returns:
            str: Human-readable string representation
        """
        return (
            f"RelationType("
            f"ID={self.ID}, name='{self.name}', "
            f"posstr='{self.posstr}', order={self.order}"
            f")"
        )

    def __repr__(self) -> str:
        """
        Detailed string representation of RelationType.

        Returns:
            str: Detailed string representation for debugging
        """
        return (
            f"RelationType(ID={self.ID}, objecttype={self.objecttype}, "
            f"PARENT_ID={self.PARENT_ID}, REVERSE_ID={self.REVERSE_ID}, "
            f"name='{self.name}', description='{self.description}', "
            f"posstr='{self.posstr}', autoreverse={self.autoreverse}, "
            f"display='{self.display}', shortcut='{self.shortcut}', "
            f"pwn='{self.pwn}', order={self.order})"
        )


class RelationTypeMapper(GeneralMapper):
    """
    Utility class for mapping between dictionary and RelationType objects.
    """

    map_obj = RelationType
