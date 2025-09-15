"""
Connector for the English Wordnet XML‑LMF file.

The connector loads all ``LexicalEntry`` and ``Synset`` elements from the
provided XML file into memory and offers simple search utilities:

* ``find_lexical_entries(lemma, pos, variant=None)`` – returns a list of
  dictionaries describing lexical entries that match the criteria.
* ``get_synsets_for_entry(entry_id)`` – returns the synset(s) associated with a
  given lexical entry.
* ``find_synsets_by_lemma(lemma, pos, variant=None)`` – combines the two
  above to return synsets directly for a lemma/part‑of‑speech pair.
"""

import os
import html

from lxml import etree
from typing import Any, Dict, List, Optional

# {"id": str, "lemma": str, "pos": str, "synset": str, "variant": Optional[str]}
LexicalEntryDict = Dict[str, Any]

# {"id": str, "definition": str, "pos": str,
# "relations": List[Dict[str, str]], "examples": List[str]}
SynsetDict = Dict[str, Any]


class EnglishWordnetConnector:
    """
    Loads an English Wordnet XML‑LMF file and provides lookup methods.

    Handle POS mapping (to be consistent with Polish Wordnet):
        5 → `'v'` (verb),
        6 → `'n'` (noun),
        7 → `'r'` (adverb),
        8 → `'a'` (adjective).
    """

    EN_WN_POS_LIST = [5, 6, 7, 8, "v", "n", "r", "a"]

    def __init__(self, xml_path: str) -> None:
        """
        Parameters
        ----------
        xml_path: str
            Full path to `en-wordnet.xml` (or any WN‑LMF file).
            Raises `FileNotFoundError` if the file does not exist.
        """
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"English Wordnet XML not found: {xml_path}")

        self.xml_path: str = xml_path

        self._synsets: List[SynsetDict] = []
        self._lexical_entries: List[LexicalEntryDict] = []

        self._load_data()

    def find_lexical_entries(
        self,
        lemma: Optional[str] = None,
        pos: Optional[str | int] = None,
        variant: Optional[str] = None,
    ) -> List[LexicalEntryDict]:
        """
        Return all lexical entries that match the given lemma, part of speech
        and optional variant. Matching is case‑insensitive.

        Parameters
        ----------
        lemma: str
            Lemma to search for.
        pos: Optional[str | int]
            Part of speech abbreviation (e.g. `n` for noun).
        variant: Optional[str]
            Optional variant attribute present in the source XML.

        Returns
        -------
        List[LexicalEntryDict]
            Matching lexical entries.
        """
        lemma_norm = lemma.lower()
        pos_norm = self._normalize_pos_if_needed(pos=pos)

        results: List[LexicalEntryDict] = []
        for entry in self._lexical_entries:
            if (
                lemma is not None
                and len(lemma)
                and entry["lemma"].lower() != lemma_norm
            ):
                continue

            if pos_norm is not None and entry["pos"].lower() != pos_norm:
                continue

            if variant is not None and entry["variant"] != variant:
                continue

            results.append(entry)

        return results

    def get_synsets_for_entry(self, entry_id: str) -> List[SynsetDict]:
        """
        Return the synset(s) associated with the lexical entry identified by
        `entry_id`. In typical LMF files there is a single synset, but a list
        is returned for completeness.

        Parameters
        ----------
        entry_id: str
            Identifier of a `LexicalEntry` (e.g. `oewn--ap-s_Gravenhage-n`).

        Returns
        -------
        List[SynsetDict]
            Associated synsets, possibly empty if the entry is unknown.
        """
        entry = next(
            (e for e in self._lexical_entries if e["id"] == entry_id),
            None,
        )
        if entry is None:
            return []

        synset_id = entry["synset"]
        synset = next((s for s in self._synsets if s["id"] == synset_id), None)

        return [synset] if synset else []

    def find_synsets_by_lemma(
        self,
        lemma: Optional[str] = None,
        pos: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[SynsetDict]:
        """
        Convenience method that finds lexical entries for a lemma/pos pair
        and then returns the linked synsets.

        Parameters
        ----------
        lemma: str
            Lemma to search for.
        pos: str
            Part of speech abbreviation.
        variant: Optional[str]
            Optional variant attribute.

        Returns
        -------
        List[SynsetDict]
            All synsets that correspond to the supplied criteria.
        """
        entries = self.find_lexical_entries(lemma, pos, variant)
        synsets: List[SynsetDict] = []

        for entry in entries:
            synsets.extend(self.get_synsets_for_entry(entry["id"]))

        return synsets

    def _load_data(self) -> None:
        """
        Parse the XML file and delegate to dedicated parsers.
        """
        tree = etree.parse(self.xml_path)
        root = tree.getroot()

        self._load_lexical_entries(root)
        self._load_synsets(root)

    def _load_lexical_entries(self, root: etree._Element) -> None:
        """
        Extract `LexicalEntry` elements and store them in `self._lexical_entries`.
        """
        for le in root.xpath("//LexicalEntry"):
            le_id = le.get("id")
            lemma_el = le.find("Lemma")
            sense_el = le.find("Sense")

            # Skip incomplete entries
            if lemma_el is None or sense_el is None:
                continue

            entry: LexicalEntryDict = {
                "id": le_id,
                "lemma": self._unescape_text(lemma_el.get("writtenForm")),
                "pos": lemma_el.get("partOfSpeech"),
                "synset": sense_el.get("synset"),
                "variant": le.get("variant"),  # optional, may be None
            }
            self._lexical_entries.append(entry)

    def _load_synsets(self, root: etree._Element) -> None:
        """
        Extract `Synset` elements and store them in `self._synsets`.
        """
        for syn in root.xpath("//Synset"):
            definition_el = syn.find("Definition")
            definition = definition_el.text if definition_el is not None else ""

            # Gather all SynsetRelation children
            relations: List[Dict[str, str]] = []
            for rel in syn.xpath("SynsetRelation"):
                relations.append(
                    {
                        "relType": rel.get("relType"),
                        "target": rel.get("target"),
                    }
                )

            # Gather all Example children
            examples: List[str] = []
            for ex in syn.xpath("Example"):
                if ex.text:
                    examples.append(ex.text)

            synset: SynsetDict = {
                "id": syn.get("id"),
                "pos": syn.get("partOfSpeech"),
                "definition": definition,
                "relations": relations,
                "examples": examples,
            }
            self._synsets.append(synset)

    @staticmethod
    def _normalize_pos_if_needed(pos: Optional[str | int] = None) -> Optional[str]:
        """
        Normalize a part‑of‑speech (POS) identifier to the WordNet POS tag.

        Parameters
        ----------
        pos : Optional[str | int], default=None
            The POS value to normalize. It can be:
            * ``None`` – the function returns ``None``.
            * a string – the value is lower‑cased and returned unchanged.
            * an integer – interpreted as a WordNet POS code where
              5 → `'v'` (verb),
              6 → `'n'` (noun),
              7 → `'r'` (adverb),
              8 → `'a'` (adjective).

        Any other integer raises ``NotImplementedError``.

        Returns
        -------
        Optional[str]
            The normalized POS tag (`'n'`, `'v'`, `'a'`, `'r'`) or the
            lower‑cased string version of the input. Returns `None` when the
            input is `None`.

        Raises
        ------
        NotImplementedError
            If `pos` is an integer that does not correspond to a known POS code.
        """

        if pos is None:
            return None

        if type(pos) == str:
            pos_norm = pos.lower()
        else:
            if pos == 8:
                pos_norm = "a"
            elif pos == 7:
                pos_norm = "r"
            elif pos == 6:
                pos_norm = "n"
            elif pos == 5:
                pos_norm = "v"
            else:
                raise NotImplementedError(f"Unknown POS {pos}")
        return pos_norm

    @staticmethod
    def _unescape_text(text: Optional[str]) -> str:
        """
        Convert XML/HTML character entity references to their literal characters.

        The WordNet LMF files sometimes contain escaped entities such as
        `&apos;`, `&quot;`, `&lt;`, `&gt;` or `&amp;` inside attribute
        values (e.g. `writtenForm="&apos;hood"`).  `html.unescape` safely
        expands all known entities to their textual representation.

        Parameters
        ----------
        text: Optional[str]
            Text possibly containing entity references.
            `None` is treated as an empty string.

        Returns
        -------
        str
            The unescaped version of *text*.
        """
        if text is None:
            return ""
        return html.unescape(text)
