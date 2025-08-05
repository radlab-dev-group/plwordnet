from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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

    @property
    def is_valid(self) -> bool:
        """
        Check if the relation type has valid basic properties.

        Returns:
            bool: True if a relation type appears valid, False otherwise
        """
        return (
            self.ID > 0
            and bool(self.name.strip())
            and self.objecttype >= 0
            and self.order >= 0
        )

    @property
    def has_parent(self) -> bool:
        """
        Check if the relation type has a parent relation.

        Returns:
            bool: True if PARENT_ID exists and is not None, False otherwise
        """
        return self.PARENT_ID is not None and self.PARENT_ID > 0

    @property
    def has_reverse(self) -> bool:
        """
        Check if the relation type has a reverse relation.

        Returns:
            bool: True if REVERSE_ID exists and is not None, False otherwise
        """
        return self.REVERSE_ID is not None and self.REVERSE_ID > 0

    @property
    def is_autoreverse(self) -> bool:
        """
        Check if the relation type is auto-reversible.

        Returns:
            bool: True if autoreverse is 1, False otherwise
        """
        return self.autoreverse == 1

    @property
    def has_description(self) -> bool:
        """
        Check if the relation type has a non-empty description.

        Returns:
            bool: True if the description exists and is not empty, False otherwise
        """
        return bool(self.description.strip())

    @property
    def has_display_name(self) -> bool:
        """
        Check if the relation type has a display name.

        Returns:
            bool: True if the display name exists and is not empty, False otherwise
        """
        return bool(self.display.strip())

    @property
    def has_shortcut(self) -> bool:
        """
        Check if the relation type has a shortcut.

        Returns:
            bool: True if the shortcut exists and is not empty, False otherwise
        """
        return bool(self.shortcut.strip())


class RelationTypeMapper:
    """
    Utility class for mapping between dictionary and RelationType objects.
    """

    @staticmethod
    def map_from_dict(data: Dict[str, Any]) -> RelationType:
        """
        Map dictionary data to a RelationType object.

        Args:
            data: Dictionary containing relation type data

        Returns:
            RelationType: Mapped RelationType object
        """
        return RelationType.from_dict(data)

    @staticmethod
    def map_from_dict_list(data_list: List[Dict[str, Any]]) -> List[RelationType]:
        """
        Map list of dictionaries to a list of RelationType objects.

        Args:
            data_list: List of dictionaries containing relation type data

        Returns:
            List[RelationType]: List of mapped RelationType objects
        """
        return [RelationType.from_dict(data) for data in data_list]

    @staticmethod
    def map_to_dict(relation_type: RelationType) -> Dict[str, Any]:
        """
        Map RelationType object to dictionary.

        Args:
            relation_type: RelationType object to map

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return relation_type.to_dict()

    @staticmethod
    def map_to_dict_list(relation_types: List[RelationType]) -> List[Dict[str, Any]]:
        """
        Map list of RelationType objects to a list of dictionaries.

        Args:
            relation_types: List of RelationType objects

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [rel_type.to_dict() for rel_type in relation_types]
