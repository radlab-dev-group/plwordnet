from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from plwordnet_handler.base.structure.elems.comment import ParsedComment


class GeneralMapper:
    """
    A static utility class for mapping operations between objects and dictionaries.

    This class provides generic mapping functionality that works with any object
    that implements the required from_dict and to_dict methods. It uses a class
    variable map_obj to determine which specific mapper implementation to use.

    Attributes:
        map_obj: The mapper object that handles the actual conversion logic
    """

    map_obj = None

    def map_from_dict(self, data: Dict[str, Any]) -> Any:
        """
        Converts a dictionary to an object using the configured mapper.

        Args:
            data (Dict[str, Any]): Dictionary containing object data

        Returns:
            Any: Object created from the dictionary data
        """

        return self.map_obj.from_dict(data)

    def map_from_dict_list(self, data_list: List[Dict[str, Any]]) -> List[Any]:
        """
        Converts a list of dictionaries to a list of objects.

        Args:
            data_list (List[Dict[str, Any]]): List of dictionaries to convert

        Returns:
            List[Any]: List of objects created from the dictionary data
        """

        return [self.map_from_dict(data=data) for data in data_list]

    @staticmethod
    def map_to_dict(map_obj: Any) -> Dict[str, Any]:
        """
        Converts the configured mapper object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the mapper object
        """

        return map_obj.to_dict()

    @staticmethod
    def map_to_dict_list(map_objects: List[Any]) -> List[Dict[str, Any]]:
        """
        Converts a list of objects to a list of dictionaries.

        Args:
            map_objects (List[Any]): List of objects to convert

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """

        return [
            GeneralMapper.map_to_dict(map_obj=map_obj) for map_obj in map_objects
        ]


@dataclass
class BaseGeneralElem(ABC):
    """
    Abstract base class for general elements in the Polish WordNet structure.

    Attributes:
        ID (int): Unique identifier for the element

    """

    ID: int


@dataclass
class GeneralElem(BaseGeneralElem, ABC):
    """
    Abstract base class for general elements in the Polish WordNet structure
    with error comment and parsed comment.

    This class defines the common attributes and interface that all elements
    must implement. It serves as a template for concrete element implementations.

    Attributes:
        status (int): Status code of the element
        owner (str): Owner identifier for the element
        error_comment (str): Error comment associated with the element
        comment (ParsedComment): Parsed comment object containing additional metadata
    """

    status: int
    owner: str
    error_comment: str
    comment: ParsedComment

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Abstract method to create an instance from dictionary data.

        Args:
            data (Dict[str, Any]): Dictionary containing element data

        Note:
            Must be implemented by concrete subclasses
        """

        pass

    @classmethod
    @abstractmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Abstract method to convert the instance to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the element

        Note:
            Must be implemented by concrete subclasses
        """

        pass


@dataclass
class GeneralRelation(ABC):
    """
    Abstract base class for general relations in the Polish WordNet structure.

    This class defines the common attributes and interface that all relation types
    must implement. It represents relationships between elements in the wordnet,
    typically connecting parent and child elements through specific relation types.

    Attributes:
        PARENT_ID (int): Identifier of the parent element in the relation
        CHILD_ID (int): Identifier of the child element in the relation
        REL_ID (int): Identifier of the relation type defining the relationship
        valid (int): Validation status of the relation (typically 0 or 1)
        owner (str): Owner identifier for the relation
    """

    PARENT_ID: int
    CHILD_ID: int
    REL_ID: int
    valid: int
    owner: str

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Abstract method to create an instance from dictionary data.

        Args:
            data (Dict[str, Any]): Dictionary containing element data

        Note:
            Must be implemented by concrete subclasses
        """

        pass

    @classmethod
    @abstractmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Abstract method to convert the instance to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the element

        Note:
            Must be implemented by concrete subclasses
        """

        pass
