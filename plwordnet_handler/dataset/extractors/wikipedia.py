import re
import json
import requests
import logging

from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote


class WikipediaExtractor:
    """
    Extractor for fetching main description content from Wikipedia articles.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_sentences: int = 3,
        prompts_dir: Optional[str] = None,
        clear_text_prompt_name: Optional[str] = None,
        openapi_configs_dir: Optional[str] = None,
    ):
        """
        Initialize Wikipedia content extractor with optional content clear.

        Args:
            timeout: Request timeout in seconds
            max_sentences: Maximum number of sentences
                           to extract from the main description
            prompts_dir : str (Optional: None)
                Directory containing prompt files;
                used by PromptHandler to load the prompt.
            clear_text_prompt_name : str (Optional: None)
                The key/name of the prompt to use
                as the system prompt for correction.
            openapi_configs_dir: str (Optional: None)
                Directory containing OpenAPI config files;
        """
        self.timeout = timeout
        self.max_sentences = max_sentences
        self.logger = logging.getLogger(__name__)

        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PlWordnet-Handler/1.0 "
                "(https://github.com/radlab-dev-group/"
                "radlab-plwordnet; pawel@radlab.dev)"
            }
        )

        self.api_handler = None
        if (
            prompts_dir is not None
            and clear_text_prompt_name is not None
            and openapi_configs_dir is not None
        ):
            self.__init_openapi_handler_cache(
                prompts_dir=prompts_dir,
                prompt_name=clear_text_prompt_name,
                openapi_configs_dir=openapi_configs_dir,
            )

    def __init_openapi_handler_cache(
        self, prompts_dir, prompt_name, openapi_configs_dir
    ):
        """
        Initialize an OpenAPI handler with caching
        for Wikipedia description extraction.

        Creates an :class:`OpenApiHandlerWithCache` instance configured
        to load prompts from `prompts_dir` using `prompt_name` and to store
        cached API responses in a dedicated work directory. `openapi_configs_dir`
        points to the directory containing OpenAPI specification files
        required by the handler.

        Args:
            prompts_dir (str): Directory containing prompt files used by the handler.
            prompt_name (str): Identifier of the prompt to be loaded.
            openapi_configs_dir (str): Directory with OpenAPI configuration files.

        The handler instance is stored in `self.api_handler` for later use.
        """
        from rdl_ml_utils.open_api.cache_api import OpenApiHandlerWithCache

        self.api_handler = OpenApiHandlerWithCache(
            prompts_dir=prompts_dir,
            prompt_name=prompt_name,
            workdir="./__cache/wikipedia_description/",
            openapi_configs_dir=openapi_configs_dir,
        )

    def extract_main_description(
        self, wikipedia_url: str, fix_content: bool = False
    ) -> Optional[str]:
        """
        Extract the main description from a Wikipedia article.

        Args:
            wikipedia_url: URL to Wikipedia article
            fix_content (bool): If True, fixes Wikipedia content (punctuation)

        Returns:
            Main description text or None if extraction failed
        """
        self.logger.debug(f"Processing article {wikipedia_url}")

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
            if fix_content and self.api_handler is not None:
                return self._fix_content(content_str=main_description)
            return main_description

        except Exception as e:
            self.logger.error(
                f"Error extracting description from {wikipedia_url}: {e}"
            )
            return None

    # def extract_multiple_descriptions(
    #     self, wikipedia_urls: list[str]
    # ) -> Dict[str, Optional[str]]:
    #     """
    #     Extract descriptions from multiple Wikipedia URLs.
    #
    #     Args:
    #         wikipedia_urls: List of Wikipedia URLs
    #
    #     Returns:
    #         Dictionary mapping URLs to their extracted descriptions
    #     """
    #     results = {}
    #
    #     for url in wikipedia_urls:
    #         try:
    #             description = self.extract_main_description(wikipedia_url=url)
    #             results[url] = description
    #             if description:
    #                 self.logger.info(
    #                     f"Successfully extracted description for: {url}"
    #                 )
    #             else:
    #                 self.logger.warning(f"Failed to extract description for: {url}")
    #         except Exception as e:
    #             self.logger.error(f"Error processing URL {url}: {e}")
    #             results[url] = None
    #     return results

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

            description = self.extract_main_description(
                wikipedia_url=wikipedia_url, fix_content=False
            )
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

    def _fix_content(self, content_str: str) -> str:
        if not len(content_str):
            return ""
        return self.api_handler.generate(text_str=content_str)

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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


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
