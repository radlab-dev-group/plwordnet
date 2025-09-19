import re
import os
import json
import logging
import requests
import datetime

from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote


class WikipediaExtractor:
    """
    Extractor for fetching main description content from Wikipedia articles.
    """

    CACHE_RESULTS = 500

    def __init__(
        self,
        timeout: int = 10,
        max_sentences: int = 3,
        cache_dir: Optional[str] = None,
        cache_results_size: Optional[int] = None,
    ):
        """
        Initialize Wikipedia content extractor with optional content clear.

        Args:
            timeout: Request timeout in seconds
            max_sentences: Maximum number of sentences
                           to extract from the main description
            cache_dir: (Optional) Directory to cache results (extracted content)
            cache_results_size: (Optional) Size of cache results (extracted content)
        """
        self.timeout = timeout
        self.max_sentences = max_sentences
        self.logger = logging.getLogger(__name__)

        self.cache_dir = cache_dir or "__cache/wikipedia/raw"
        os.makedirs(self.cache_dir, exist_ok=True)

        self._cached_content = {}
        self._content_batch = {}
        self.cache_results_size = cache_results_size or self.CACHE_RESULTS

        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PlWordnet-Handler/1.0 "
                "(https://github.com/radlab-dev-group/"
                "radlab-plwordnet; pawel@radlab.dev)"
            }
        )

        loaded_cnt = self._load_cache_from_workdir()
        if loaded_cnt:
            print(f" ~> loaded {loaded_cnt} cached items from {self.cache_dir}")

    def extract_main_description(self, wikipedia_url: str) -> Optional[str]:
        """
        Extract the main description from a Wikipedia article.

        Args:
            wikipedia_url: URL to Wikipedia article

        Returns:
            Main description text or None if extraction failed
        """
        self.logger.debug(f"Processing article {wikipedia_url}")
        _cached = self._cached_content.get(wikipedia_url)
        if _cached is not None:
            return _cached

        try:
            article_title = self._extract_article_title(wikipedia_url=wikipedia_url)
            if not article_title:
                self.logger.error(
                    f"Could not extract article title from URL: {wikipedia_url}"
                )
                return None

            language = self._extract_language_from_url(wikipedia_url=wikipedia_url)
            if not language:
                self.logger.error(
                    f"Could not determine language from URL: {wikipedia_url}"
                )
                return None

            content = self._fetch_article_content(
                article_title=article_title, language=language
            )
            if not content:
                return None

            main_description = self._extract_and_clean_description(content=content)

            self._cached_content[wikipedia_url] = main_description
            self._content_batch[wikipedia_url] = main_description
            if len(self._content_batch) >= self.cache_results_size:
                self._store_cache_batch(batch=self._content_batch)
                self._content_batch.clear()

            return main_description

        except Exception as e:
            self.logger.error(
                f"Error extracting description from {wikipedia_url}: {e}"
            )
            return None

    def get_article_info(self, wikipedia_url: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a Wikipedia article.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Dictionary with article information or None if failed
        """
        try:
            article_title = self._extract_article_title(wikipedia_url=wikipedia_url)
            language = self._extract_language_from_url(wikipedia_url=wikipedia_url)
            if not article_title or not language:
                return None

            description = self.extract_main_description(wikipedia_url=wikipedia_url)
            return {
                "url": wikipedia_url,
                "title": article_title,
                "language": language,
                "description": description,
                "is_valid": description is not None,
            }

        except Exception as e:
            self.logger.error(f"Error getting article info for {wikipedia_url}: {e}")
            return None

    def close(self):
        """
        Close the session.
        """
        if self.session:
            self.session.close()

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences.

        Simple sentence splitting can be improved with more sophisticated methods.
        Handle common abbreviations that shouldn't trigger sentence breaks.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        text = re.sub(
            r"\b(np|tzn|tj|itp|itd|por|zob|ang|łac|gr|fr|niem|ros)\.\s*",
            r"\1._ABBREV_",
            text,
        )
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.replace("_ABBREV_", ".") for s in sentences]
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove parenthetical notes at the beginning (common in Wikipedia)
        text = re.sub(r"^\([^)]*\)\s*", "", text)
        # Clean up common Wikipedia artifacts
        #  - remove reference markers
        text = re.sub(r"\[.*?\]", "", text)
        #  - remove template markers
        text = re.sub(r"\{.*?\}", "", text)
        # Normalize punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip()

    @staticmethod
    def is_valid_wikipedia_url(url: str) -> bool:
        """
        Check if URL is a valid Wikipedia URL.

        Args:
            url: URL to validate

        Returns:
            True if valid Wikipedia URL, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            return "wikipedia.org" in parsed_url.netloc and (
                parsed_url.path.startswith("/wiki/") or "title=" in parsed_url.query
            )
        except Exception:
            return False

    def _extract_article_title(self, wikipedia_url: str) -> Optional[str]:
        """
        Extract article title from Wikipedia URL.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Article title or None if extraction failed
        """
        try:
            parsed_url = urlparse(wikipedia_url)

            if "/wiki/" in parsed_url.path:
                # Standard format:
                #  https://pl.wikipedia.org/wiki/Article_Title
                title = parsed_url.path.split("/wiki/")[-1]
                return unquote(title)
            elif "title=" in parsed_url.query:
                # Query format:
                #  https://pl.wikipedia.org/w/index.php?title=Article_Title
                query_params = parse_qs(parsed_url.query)
                if "title" in query_params:
                    return query_params["title"][0]
            return None
        except Exception as e:
            self.logger.error(
                f"Error extracting title from URL {wikipedia_url}: {e}"
            )
            return None

    def _extract_language_from_url(self, wikipedia_url: str) -> Optional[str]:
        """
        Extract language code from Wikipedia URL.

        Args:
            wikipedia_url: Wikipedia URL

        Returns:
            Language code (e.g., 'pl', 'en') or None if extraction failed
        """
        try:
            parsed_url = urlparse(wikipedia_url)
            if "wikipedia.org" in parsed_url.netloc:
                parts = parsed_url.netloc.split(".")
                if len(parts) >= 3 and parts[1] == "wikipedia":
                    return parts[0]
            return None
        except Exception as e:
            self.logger.error(
                f"Error extracting language from URL {wikipedia_url}: {e}"
            )
            return None

    def _fetch_article_content(
        self, article_title: str, language: str
    ) -> Optional[str]:
        """
        Fetch article content using Wikipedia API.

        Args:
            article_title: Title of the Wikipedia article
            language: Language code

        Returns:
            Article content or None if fetch failed
        """
        try:
            api_url = f"https://{language}.wikipedia.org/w/api.php"

            # API parameters to get article content
            params = {
                "action": "query",
                "format": "json",
                "titles": article_title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsectionformat": "plain",
            }

            response = self.session.get(api_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                self.logger.warning(f"No pages found for title: {article_title}")
                return None

            # Get the first (and should be only) a page
            page_id = next(iter(pages))
            page_data = pages[page_id]
            if "missing" in page_data:
                self.logger.warning(f"Page not found: {article_title}")
                return None
            extract = page_data.get("extract", "")
            if not extract:
                self.logger.warning(f"No extract found for: {article_title}")
                return None
            return extract
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching article {article_title}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for article {article_title}: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching article {article_title}: {e}"
            )
            return None

    def _extract_and_clean_description(self, content: str) -> str:
        """
        Extract and clean the main description from article content.

        Args:
            content: Raw article content

        Returns:
            Cleaned main description
        """
        if not content:
            return ""

        sentences = self._split_into_sentences(text=content)
        main_sentences = sentences[: self.max_sentences]
        description = " ".join(main_sentences)
        description = self._clean_text(text=description)

        return description

    def _store_cache_batch(self, batch):
        """
        Persist the current in-memory batch to a timestamped JSON file.

        This method serializes the provided key-value pairs (input text -> corrected
        text) to a file located in the configured working directory. The output
        filename encodes a high-resolution timestamp:
          {YYYYMMDD_HHMMSS}.{microseconds}.json

        Notes
        -----
        - The working directory is expected to exist (created during initialization).
        - JSON is written with indentation (2 spaces), and ensure_ascii=False to
          preserve non-ASCII characters.
        - Thread-safety: callers should guard concurrent invocations (e.g., via
          `self.lock_map`) to avoid interleaved writes.

        Parameters
        ----------
        batch : dict
            Mapping of source strings to their corrected results to be persisted.

        Returns
        -------
        None
        """
        date_now = datetime.datetime.now()
        data_as_str = date_now.strftime("%Y%m%d_%H%M%S")
        data_as_str += f".{date_now.microsecond:06d}"
        out_f_path = os.path.join(self.cache_dir, f"{data_as_str}.json")

        with open(out_f_path, "w") as f:
            print(f" ~> storing content to cache file: {out_f_path}")
            json.dump(batch, f, indent=2, ensure_ascii=False)

    def _load_cache_from_workdir(self) -> int:
        """
        Scan workdir for .json cache files and load their contents
        into self._correct_texts.

        Returns
        -------
        int
            Number of unique items loaded into the in-memory cache.
        """
        loaded = 0
        workdir_content = sorted(os.listdir(self.cache_dir))
        for f_name in workdir_content:
            if not f_name.lower().endswith(".json"):
                continue

            fpath = os.path.join(self.cache_dir, f_name)
            if not os.path.isfile(fpath):
                continue

            with open(fpath, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    self._cached_content[k] = v
                    loaded += 1
        return loaded

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        if self._content_batch is None or not len(self._content_batch):
            return
        self._store_cache_batch(batch=self._content_batch)
        self._content_batch.clear()


def is_wikipedia_url(url: str) -> bool:
    """
    Utility function to check if URL is a Wikipedia URL.

    Args:
        url: URL to check

    Returns:
        True if Wikipedia URL, False otherwise
    """
    return WikipediaExtractor().is_valid_wikipedia_url(url)


def extract_wikipedia_description(url: str, max_sentences: int = 3) -> Optional[str]:
    """
    Utility function to extract Wikipedia description.

    Args:
        url: Wikipedia URL
        max_sentences: Maximum number of sentences to extract

    Returns:
        Main description or None if extraction failed
    """
    if not is_wikipedia_url(url=url):
        return None

    with WikipediaExtractor(max_sentences=max_sentences) as extractor:
        return extractor.extract_main_description(url)
