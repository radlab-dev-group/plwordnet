from plwordnet_handler.dataset.extractors.wikipedia import WikipediaExtractor

url = "https://pl.wikipedia.org/wiki/Diana_(1899)"

with WikipediaExtractor(max_sentences=10) as extractor:
    info = extractor.get_article_info(url)
    print(f"Title: {info['title']}")
    print(f"Description: {info['description']}")
