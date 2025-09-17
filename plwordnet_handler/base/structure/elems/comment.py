import re
import logging

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


class ExampleType(Enum):
    """
    Enum for different types of usage examples
    """

    STANDARD_P = "P"
    STANDARD_D = "D"
    STANDARD_W = "W"
    KPWR = "KPWr"
    UNKNOWN = "UNKNOWN"


@dataclass
class SentimentAnnotation:
    """
    Data class representing a sentiment annotation (##A1, ##A2, etc.)
    """

    annotation_id: str  # e.g., "A1", "A2"
    emotions: List[str]  # e.g., ["smutek", "złość"]
    categories: List[str]  # e.g., ["błąd"]
    strength: str  # e.g., '+/- s' or '+/- w' or 'amb'
    example: str  # example sentence in brackets

    def as_dict(self):
        return {
            "annotation_id": self.annotation_id,
            "emotions": self.emotions,
            "categories": self.categories,
            "strength": self.strength,
            "example": self.example,
        }


@dataclass
class UsageExample:
    """
    Data class representing a usage example with type information
    """

    text: str
    source_pattern: Optional[str] = None
    example_type: Optional[ExampleType] = None

    def as_dict(self) -> Dict[str, str]:
        return {
            "text": self.text,
            # "example_type": self.example_type.value,
            "source_pattern": self.source_pattern,
        }


@dataclass
class ExternalUrlDescription:
    """
    Data class representing an external URL description (##L)
    """

    url: str
    content: str = None

    def as_dict(self):
        return {"url": self.url, "content": self.content}


@dataclass
class ParsedComment:
    """
    Data class representing the parsed comment structure
    """

    original_comment: str
    base_domain: Optional[str] = None  # ##K content
    definition: Optional[str] = None  # ##D content
    usage_examples: List[UsageExample] = None  # All examples from [ ]
    external_url_description: Optional[ExternalUrlDescription] = None  # ##L content
    sentiment_annotations: List[SentimentAnnotation] = None  # ##A1, ##A2, etc.

    def __post_init__(self):
        if self.usage_examples is None:
            self.usage_examples = []
        if self.sentiment_annotations is None:
            self.sentiment_annotations = []

    def as_dict(self):
        return {
            "original_comment": self.original_comment,
            "base_domain": self.base_domain,
            "definition": self.definition,
            "usage_examples": [e.as_dict() for e in self.usage_examples],
            "external_url_description": (
                self.external_url_description.as_dict()
                if self.external_url_description
                else {}
            ),
            "sentiment_annotations": (
                [s.as_dict() for s in self.sentiment_annotations]
                if self.sentiment_annotations
                else []
            ),
        }


class CommentParser:
    """
    Parser for plWordnet comment annotations with sentiment analysis.
    """

    MIN_EXAMPLE_LENGTH = 20
    PHRASES_CANNOT_FIND = ["brak danych", "."]
    PHRASES_NO_TEXTUAL_DATA = ["brak danych <"]
    STRIP_CHARS_FROM_TEXTUAL_DATA = ["<", ">", "[", "]", "{", "}", ":", "#", " "]
    REMOVE_PHRASES = [
        "##\nW:",
        "##P:",
        "##:P:",
        "##W':",
        "##A1",
        "##A2",
        "##A3",
        "##Bełza:",
        "##Dołęga:",
        "##Leśmian:",
        "##Bałucki:",
        "##Żeromski:",
        "##Jasieński:",
        "##Żuławski:",
        "##Dąbrowska:",
        "##Grabiński:",
        "##Mniszkówna:",
        "##Wyspiański:",
        "##Dygasiński:",
        "##Łuszczewska:",
        "##Żurakowska:",
        "##Gąsiorowski:",
        "##Chocieszyński:",
        "##May-Winnetou:",
        "##Słowacki:",
        "##Łoziński:",
        "##Zabłocki",
        "##Marrené-Balzac:",
        "##Siemieński:",
        "##Baliński:",
        "##Zegadłowicz:",
        "##Krasiński:",
        "##Miałaczewski:",
        "##Kołaczkowski:",
        "##Jan-z-Czarnkowa:",
        "##Pascal-Boy:",
        "##Świętochowski:",
        "##Rodziewiczówna:",
        "##Swift-Nepomucen:",
        "##Poincare-Horwitz:",
        "##Dostojewski-Beaupré",
        "##Laclos-Żeleński:",
        "##Dołęga-Mostowicz:",
        "##Conrad-Zagórska:",
        "##Wallace-Stefański:",
        "##Ćwierczakiewiczowa:",
        "##Szekspir-Paszkowski:",
        "##Wassermann-Mirandola:",
        "## MSZ:",
        "{##L:",
        "##K:",
        "##L:",
        "##D:",
        "#P:",
        "##A1:",
        "##A2:",
        "##A3:",
        "#W:",
        "#Ko:",
        "##Ko",
        "##P",
        "#P",
        "http://pl.wikipedia.org/wiki",
        "https://pl.wikipedia.org/wiki",
    ]

    def __init__(self):
        # Regex patterns for different comment elements
        self.base_domain_pattern = r"#[#]?K:\s*([^#]+?)(?=\s*##|$|\.)"
        self.definition_pattern = r"#[#]?[DPW][':]?\s*([^#\[{]+?)(?=\s*\[|##|{|$)"
        # self.external_url_pattern = r"{##L:\s*([^}]+?)}"
        self.external_url_pattern = r"{##L:\s*([^}]+?)(?:\s|})"

        # Sentiment annotation pattern to capture strength
        self.sentiment_annotation_pattern = (
            r"##(A\d+):\s*\{([^}]+)\}\s*([+-]\s*[sm]|amb)\s*\[([^\]]+)\]"
        )

        # Pattern to find all bracketed examples (excluding sentiment annotations)
        self.bracketed_example_pattern = r"\[([^\]]+?)\]"

        # Pattern to identify the example type within brackets
        self.example_type_pattern = r"##([A-Za-z0-9]+):\s*(.+)"

        self.logger = logging.getLogger(__name__)

    def parse_comment(self, comment: str) -> ParsedComment:
        """
        Parse a plWordnet comment string into structured data.

        Args:
            comment: Raw comment string from database

        Returns:
            ParsedComment: Parsed comment structure
        """
        if not comment or not comment.strip():
            return ParsedComment(original_comment=comment)

        parsed = ParsedComment(original_comment=comment)
        parsed.base_domain = self._extract_base_domain(comment=comment)
        parsed.definition = self._extract_definition(comment=comment)
        parsed.sentiment_annotations = self._extract_sentiment_annotations(
            comment=comment
        )

        parsed.usage_examples = self._extract_bracketed_examples(
            comment=comment, sentiment_annotations=parsed.sentiment_annotations
        )

        parsed.external_url_description = self._extract_external_url_description(
            comment=comment
        )

        return parsed

    @staticmethod
    def _parse_emotions_and_categories(content: str) -> Tuple[List[str], List[str]]:
        """
        Parse emotions and categories from the content inside {}.

        Expected format: "emotion1, emotion2; category1, category2"
        """
        emotions = []
        categories = []

        if ";" in content:
            # Split by semicolon to separate emotions from categories
            parts = content.split(";", 1)
            emotions_part = parts[0].strip()
            categories_part = parts[1].strip() if len(parts) > 1 else ""

            # Parse emotions
            if emotions_part:
                emotions = [emotion.strip() for emotion in emotions_part.split(",")]

            # Parse categories
            if categories_part:
                categories = [
                    category.strip() for category in categories_part.split(",")
                ]
        else:
            # If no semicolon, treat everything as emotions
            emotions = [emotion.strip() for emotion in content.split(",")]

        return emotions, categories

    def _extract_base_domain(self, comment: str) -> Optional[str]:
        """
        Extract base_domain from ##K tag.
        """
        match = re.search(self.base_domain_pattern, comment)
        if match:
            domain_str = match.group(1).strip()
            if not domain_str.endswith("."):
                domain_str += "."
            return self.__clear_textual_data(text=domain_str, min_len=2)
        return None

    def _extract_definition(self, comment: str) -> Optional[str]:
        """
        Extract the definition part of a plWordnet comment.

        The standard format places the definition after a `##D`, `##W` or
        `##S` tag, e.g.:

            ##D: a small, domesticated carnivorous mammal

        In practice many comments omit those tags – the definition is simply the
        raw text of the comment.  When the regular `definition_pattern` does
        **not** match, we now treat the whole comment as a candidate definition
        and run it through the same cleaning/validation pipeline that a tagged
        definition would use.

        Returns
        -------
        Optional[str]
            * The cleaned definition string when a tag is found **or** when the
              fallback succeeds.
            * `None` when the comment is empty, consists only of noise
              (e.g. “brak danych”), or is shorter than `MIN_EXAMPLE_LENGTH`.
        """
        # Try regex “##D/##W/##S” extraction.
        match = re.search(self.definition_pattern, comment)
        if match:
            return self.__clear_textual_data(text=match.group(1).strip())

        # No regex tag -> fall back to the entire comment.
        #    This is useful for legacy entries that never received a tag.
        #    We still apply the same sanitising steps so that the result
        #    respects the project's quality rules.
        cleaned = self.__clear_textual_data(text=comment.strip())
        if cleaned:
            return cleaned

        # 3️⃣ Nothing usable was found.
        return None

    def _extract_sentiment_annotations(
        self, comment: str
    ) -> List[SentimentAnnotation]:
        """
        Extract sentiment annotations from ##A1, ##A2, etc. tags.
        Format: ##A1: {emotions; categories} - strength [example]
        """
        annotations = []
        matches = re.findall(self.sentiment_annotation_pattern, comment)

        for match in matches:
            annotation_id = match[0]  # A1, A2, etc.
            emotions_and_categories = match[1]  # content inside {}
            strength = match[2]  # s, w, etc.

            # content inside []
            example = self.__clear_textual_data(text=match[3])
            if not example or not len(example):
                continue

            # Parse emotions and categories from the {} content
            emotions, categories = self._parse_emotions_and_categories(
                content=emotions_and_categories
            )

            annotation = SentimentAnnotation(
                annotation_id=annotation_id,
                emotions=emotions,
                categories=categories,
                strength=strength,
                example=example,
            )
            annotations.append(annotation)

        return annotations

    def _extract_bracketed_examples(
        self, comment: str, sentiment_annotations: List[SentimentAnnotation]
    ) -> List[UsageExample]:
        """
        Extract usage examples from [ ] brackets,
        excluding those already captured in sentiment annotations.

        Args:
            comment: Raw comment string
            sentiment_annotations: Already parsed sentiment annotations to exclude

        Returns:
            List of UsageExample objects with type information
        """
        examples = []
        sentiment_example_texts = {
            annotation.example for annotation in sentiment_annotations
        }
        bracketed_matches = re.findall(self.bracketed_example_pattern, comment)
        for bracketed_content in bracketed_matches:
            # Skip if this is a sentiment annotation example
            if bracketed_content.strip() in sentiment_example_texts:
                continue

            example_type, source_pattern, example_text = (
                self._determine_example_type(bracketed_content=bracketed_content)
            )
            example_text = self.__clear_textual_data(text=example_text)
            if not example_text or not len(example_text):
                continue

            example = UsageExample(
                text=example_text,
                example_type=example_type,
                source_pattern=source_pattern,
            )
            examples.append(example)
        return examples

    def _determine_example_type(
        self, bracketed_content: str
    ) -> Tuple[ExampleType, Optional[str], str]:
        """
        Determine the type of example based on the ##STR: pattern inside brackets.

        Args:
            bracketed_content: Content inside [ ] brackets

        Returns:
            Tuple of (ExampleType, source_pattern, actual_example_text)
        """
        match = re.match(self.example_type_pattern, bracketed_content)

        if match:
            pattern_str = match.group(1)
            example_text = match.group(2).strip()
            example_text = self.__clear_textual_data(text=example_text)
            if pattern_str == ExampleType.STANDARD_D:
                return ExampleType.STANDARD_D, f"##{pattern_str}", example_text
            elif pattern_str == ExampleType.STANDARD_W:
                return ExampleType.STANDARD_W, f"##{pattern_str}", example_text
            elif pattern_str == ExampleType.STANDARD_P:
                return ExampleType.STANDARD_P, f"##{pattern_str}", example_text
            elif pattern_str == ExampleType.KPWR:
                return ExampleType.KPWR, f"##{pattern_str}", example_text
            else:
                return ExampleType.UNKNOWN, f"##{pattern_str}", example_text
        else:
            return ExampleType.UNKNOWN, None, bracketed_content.strip()

    def _extract_external_url_description(
        self, comment: str
    ) -> Optional[ExternalUrlDescription]:
        """
        Extract external URL description from ##L tag.

        Args:
            comment: Raw comment string

        Returns:
            ExternalUrlDescription: extracted URL or None
        """
        match = re.search(self.external_url_pattern, comment)
        if match:
            url = match.group(1).strip()
            if url and len(url):
                url = url.replace("http://", "https://")
            return ExternalUrlDescription(url=url)
        return None

    def __clear_textual_data(
        self, text: str | None, min_len: Optional[int] = None
    ) -> Optional[str]:
        """
        Clean and normalize textual data by removing unwanted phrases and characters.

        This private method performs text cleaning operations by:
        1. Removing predefined phrases from REMOVE_PHRASES
        2. Stripping specific characters from STRIP_CHARS_FROM_TEXTUAL_DATA
        3. Applying additional validation through __return_str_or_none

        Args:
            text: Input text string to be cleaned, or None
            min_len: (Optional) Minimum length of text to validate
            if not given, then default `self.MIN_EXAMPLE_LENGTH` will be used.

        Returns:
            Optional[str]: Cleaned text string, or None if input is empty/invalid

        Note:
            Uses class constants REMOVE_PHRASES and STRIP_CHARS_FROM_TEXTUAL_DATA
            for cleaning operations.
        """

        if not text:
            return None

        for c in self.REMOVE_PHRASES:
            text = text.replace(c, "").strip()

        for c in self.STRIP_CHARS_FROM_TEXTUAL_DATA:
            text = text.strip(c)

        return self.__return_str_or_none(text=text.strip(), min_len=min_len)

    def __return_str_or_none(
        self, text: str, min_len: Optional[int] = None
    ) -> Optional[str]:
        """
        Validate and filter text based on predefined criteria.

        This private method applies multiple validation checks:
        1. Returns None for empty strings
        2. Checks against phrases in PHRASES_NO_TEXTUAL_DATA (partial matches)
        3. Checks against phrases in PHRASES_CANNOT_FIND (exact matches)
        4. Validates minimum length against MIN_EXAMPLE_LENGTH

        Args:
            text: Input text string to validate
            min_len: (Optional) Minimum length of text to validate
            if not given, then default `self.MIN_EXAMPLE_LENGTH` will be used.

        Returns:
            Optional[str]: Original text if it passes all validation checks,
                          None if text fails any validation criteria

        Note:
            Uses class constants PHRASES_NO_TEXTUAL_DATA, PHRASES_CANNOT_FIND,
            and MIN_EXAMPLE_LENGTH for validation.
        """

        if not len(text):
            return None
        text = text.strip()
        for ph in self.PHRASES_NO_TEXTUAL_DATA:
            if ph in text:
                return None
        for pg in self.PHRASES_CANNOT_FIND:
            if pg == text:
                return None

        min_len = min_len if min_len else self.MIN_EXAMPLE_LENGTH
        if len(text) < min_len:
            return None

        return text


def parse_plwordnet_comment(comment: str) -> ParsedComment:
    """
    Convenience function to parse a plWordnet comment.

    Args:
        comment: Raw comment string

    Returns:
        ParsedComment: Parsed comment structure
    """
    parser = CommentParser()
    p_comment = parser.parse_comment(comment)

    # import json
    # print(json.dumps(p_comment.as_dict(), indent=2, ensure_ascii=False))

    return p_comment
