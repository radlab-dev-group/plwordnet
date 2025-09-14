from plwordnet_handler.base.connectors.en_wordnet import EnglishWordnetConnector

wn_path = "/mnt/data2/data/lexicons/english-wordnet-2024.xml/wn.xml"

connector = EnglishWordnetConnector(wn_path)

entries = connector.find_lexical_entries("hood", "n")
print("Found entries:", entries)

if entries:
    synsets = connector.get_synsets_for_entry(entries[0]["id"])
    print("Synsets for entry:", synsets)

synsets = connector.find_synsets_by_lemma("hood", "n")
print("Synsets by lemma:", synsets)
