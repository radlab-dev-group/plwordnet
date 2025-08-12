# PLWordNet Handler

---

**Wersja: 1.0.0**

Kompleksowa biblioteka Python do pracy z polskim słownikiem semantycznym Słowosieć. 
Zapewnia wysokopoziomowy interfejs API do pobierania jednostek leksykalnych, 
relacji semantycznych, analizy komentarzy oraz eksportu danych do różnych formatów.

**Repository:** https://github.com/radlab-dev-group/radlab-plwordnet

---

## Opis

PLWordNet Handler to zaawansowana biblioteka umożliwiająca łatwy dostęp do danych 
z polskiej Słowosieci (PLWordNet). Biblioteka oferuje bogaty zestaw narzędzi do 
pracy z jednostkami leksykalnymi, relacjami semantycznymi, synsetami oraz dodatkowymi 
informacjami takimi jak komentarze z anotacjami sentymentalnymi i przykłady użycia.

### Główne cechy:
- **Elastyczne połączenia z bazami danych** - wsparcie dla MySQL z możliwością rozszerzenia
- **Konwersja do grafów NetworkX** - eksport danych do formatów grafowych
- **Zaawansowane parsowanie komentarzy** - analiza komentarzy z anotacjami sentymentalnymi
- **Integracja z Wikipedią** - automatyczne wzbogacanie danych o opisy z Wikipedii
- **Narzędzia do tworzenia zbiorów danych** - eksport danych do uczenia modeli embeddingów
- **Aplikacje konsolowe** - gotowe narzędzia CLI do pracy z danymi

---

## Wymagania

- Python 3.6 lub nowszy
- Dostęp do bazy danych PLWordNet (MySQL)
- Zależności wymienione w `requirements.txt`

---

## Instalacja

### Podstawowa instalacja

``` bash
pip install git+https://github.com/radlab-dev-group/radlab-plwordnet.git
```

### Instalacja deweloperska

``` bash
# Klonowanie repozytorium
git clone https://github.com/radlab-dev-group/radlab-plwordnet.git

# Instalacja w trybie deweloperskim
cd radlab-plwordnet
pip install -e .
```

Po instalacji będzie dostępne narzędzie CLI `plwordnet-cli`.

### Instalacja predefiniowanych grafów

Informacje odnośnie procesu instalacji grafów. Proces jest automatyczny:
1. **Opcje instalacji**:
    - `PLWORDNET_DOWNLOAD_TEST=1`: pobiera graf testowy Słowosieci
    - `PLWORDNET_DOWNLOAD_FULL=1`: pobiera pełen graf Słowosieci

2. **Automatyczne pobieranie i rozpakowywanie pobranych grafów**:
    - Pobiera pliki grafów w formacie `.gz`
    - Rozpakowuje je do katalogu `INSTALL_DIR/plwordnet_handler/resources/graphs/`
    - Usuwa tymczasowe pliki `.gz` po rozpakowaniu

3. **Obsługa błędów**: 
    - Wyświetlane są komunikaty o postępie
    - Obsłużone są błędy pobierania

``` shell
# Instalacja z grafami
PLWORDNET_DOWNLOAD_FULL=1 pip install .

# Jeśli nie działa, spróbuj z verbose
PLWORDNET_DOWNLOAD_FULL=1 pip install -v .

# Lub develop mode
PLWORDNET_DOWNLOAD_FULL=1 pip install -e .

# Instalacja testowych i pełnych grafów z usuwaniem post install
export PLWORDNET_DOWNLOAD_TEST=1 
export PLWORDNET_DOWNLOAD_FULL=1 
pip install . && rm -rf build *.egg-info plwordnet_handler/resources 
```

---

## Konfiguracja

### Konfiguracja bazy danych

Utwórz plik konfiguracyjny dla bazy danych (np. `db-config.json`):

``` json
{ 
    "host": "localhost", 
    "port": 3306, 
    "user": "username", 
    "password": "password", 
    "database": "plwordnet_db" 
}
```

---

## Użycie

### Podstawowe operacje z API

``` python
from plwordnet_handler.connectors.db_connector import PlWordnetAPIMySQLDbConnector

# Utworzenie połączenia z bazą danych
connector = PlWordnetAPIMySQLDbConnector("db-config.json")

# Użycie context managera
with connector as api: 
    # Pobieranie jednostek leksykalnych 
    lexical_units = api.get_lexical_units(limit=10)
    
    # Pobieranie synsetów
    synsets = api.get_synsets(limit=5)
    
    # Pobieranie relacji leksykalnych
    lex_relations = api.get_lexical_relations(limit=10)
    
    # Przetwarzanie wyników
    for unit in lexical_units:
        print(f"Lemma: {unit.lemma}, POS: {unit.pos}")
        print(f"Definition: {unit.comment.definition}")
```

### Praca z komentarzami i anotacjami

``` python
from plwordnet_handler.api.data.comment import parse_plwordnet_comment

# Przykład komentarza z PLWordNet
comment_text = "##D: człowiek zajmujący się nauką ##A1: {nauka; pozytywne} - s [naukowiec pracuje w laboratorium]"

# Parsowanie komentarza
parsed = parse_plwordnet_comment(comment_text)
print(f"Definicja: {parsed.definition}") 
print(f"Liczba anotacji sentymentalnych: {len(parsed.sentiment_annotations)}") 
print(f"Przykłady użycia: {len(parsed.usage_examples)}")

# Dostęp do szczegółów anotacji
for annotation in parsed.sentiment_annotations: 
    print(f"Emocje: {annotation.emotions}") 
    print(f"Kategorie: {annotation.categories}") 
    print(f"Siła: {annotation.strength}")
```

### Konwersja do grafów NetworkX

``` python
from plwordnet_handler.structure.polishwordnet import PolishWordnet
from plwordnet_handler.connectors.db.db_to_nx import GraphMapper

# Utworzenie instancji PolishWordnet
with PolishWordnet(db_config_path="db-config.json") as pwn:
    # Utworzenie mappera grafów 
    mapper = GraphMapper(polish_wordnet=pwn)

    # Konwersja do grafu synsetów
    synset_graph = mapper.convert_to_synset_graph(limit=100)

    # Konwersja do grafu jednostek leksykalnych
    lexical_graph = mapper.convert_to_lexical_unit_graph(limit=100)

    # Kombinowany graf synsetów i jednostek
    combined_graph = mapper.convert_to_synset_with_units_graph(limit=100)

    print(f"Graf synsetów: {synset_graph.number_of_nodes()} węzłów, {synset_graph.number_of_edges()} krawędzi")
```

### Integracja z Wikipedią

``` python
from plwordnet_handler.external.wikipedia import WikipediaExtractor

# Automatyczne wzbogacanie danych o artykuły z Wikipedii
with WikipediaExtractor(max_sentences=3) as extractor: 
    url = "[https://pl.wikipedia.org/wiki/Informatyka](https://pl.wikipedia.org/wiki/Informatyka)"
    info = extractor.get_article_info(url)

    print(f"Tytuł: {info['title']}")
    print(f"Opis: {info['description']}")
```

---

## Narzędzia CLI

### Podstawowe użycie plwordnet-cli

``` bash
# Podstawowa konwersja bazy do grafów NetworkX
plwordnet-cli
    --db-config resources/db-config.json
    --convert-to-nx-graph
    --nx-graph-dir resources/graphs
    --log-level INFO
```

### Eksport z integracją Wikipedii

``` bash
# Konwersja z pobieraniem artykułów Wikipedii (może być bardzo czasochłonne!)
plwordnet-cli
    --db-config resources/db-config.json
    --convert-to-nx-graph
    --nx-graph-dir resources/graphs
    --extract-wikipedia-articles
    --limit 1000
```

### Dostępne opcje CLI

- `--db-config` - ścieżka do pliku konfiguracji bazy danych
- `--convert-to-nx-graph` - konwersja danych do grafów NetworkX
- `--nx-graph-dir` - katalog do zapisu grafów NetworkX
- `--extract-wikipedia-articles` - pobieranie artykułów z Wikipedii
- `--limit` - ograniczenie liczby przetwarzanych rekordów
- `--log-level` - poziom logowania (DEBUG, INFO, WARNING, ERROR, CRITICAL)

---

## Tworzenie zbiorów danych do embeddings

`Biblioteka zawiera narzędzia` do tworzenia zbiorów danych do uczenia modeli
embeddingów semantycznych:
``` python
from plwordnet_dataset.exporter.embedder import WordnetToEmbedderConverter

# Tworzenie konwertera
converter = WordnetToEmbedderConverter(
    xlsx_path="resources/relation-types-weights.xlsx",  # plik z wagami relacji
    graph_path="resources/graphs",                      # katalog z grafami NetworkX
    init_converter=True                                 # inicjalizacja przy tworzeniu
)

# Eksport zbioru danych
success = converter.export(
    output_file="embedder_dataset.jsonl",
    limit=10000,                    # ograniczenie liczby próbek
    low_high_ratio=2.0             # stosunek relacji o niskich do wysokich wag
)

if success:
    print("Eksport zakończony sukcesem")
else:
    print("Wystąpił błąd podczas eksportu")

``` 

Aplikacja CLI dla eksportu zbiorów danych:
``` bash
python embedder_dataset_dump.py
    --xlsx-weights resources/relation-types-weights.xlsx
    --graph-path resources/graphs
    --output embedder_dataset.jsonl
    --limit 10000
```

---

## Główne moduły

### `plwordnet_handler.api`
- **`plwordnet_i.py`** - abstrakcyjny interfejs API
- **`data/`** - struktury danych (jednostki leksykalne, synsets, komentarze)

### `plwordnet_handler.connectors`
- **`connector_i.py`** - interfejs dla połączeń
- **`db_connector.py`** - implementacja połączeń MySQL
- **`mysql.py`** - niskopoziomowe operacje MySQL

### `plwordnet_handler.structure`
- **`polishwordnet.py`** - główna klasa do pracy z PLWordNet
- **`elems/`** - elementy struktury (synsets, jednostki leksykalne, relacje)

### `plwordnet_handler.external`
- **`wikipedia.py`** - integracja z Wikipedią

### `plwordnet_dataset`
- **`exporter/embedder.py`** - eksport danych do uczenia embeddingów

---

## Struktury danych

### Jednostka leksykalna (LexicalUnit)

``` python
# Główne atrybuty jednostki leksykalnej
unit.lemma # lemma
unit.pos # część mowy
unit.domain # domena
unit.comment # sparsowany komentarz 
unit.status # status jednostki
``` 

### Synset (Synset)

``` python
# Główne atrybuty synsetu

synset.ID # identyfikator 
synset.definition # definicja 
synset.unitsstr # ciąg jednostek 
synset.comment # sparsowany komentarz 
synset.isabstract # czy abstrakcyjny
``` 

### Sparsowany komentarz (ParsedComment)

``` python
# Elementy komentarza
comment.definition # definicja 
comment.usage_examples # przykłady użycia 
comment.sentiment_annotations # anotacje sentymentalne 
comment.external_url_description # opisy z zewnętrznych URL
``` 

---

## Przykłady zaawansowanego użycia

### Analiza relacji semantycznych
``` python

with connector as api:
    # Pobieranie typów relacji relation_types = api.get_relation_types()
    # Pobieranie relacji między synsetami
    synset_relations = api.get_synset_relations(limit=100)
    
    # Analiza relacji
    for relation in synset_relations:
        if relation.is_active:
            print(f"Relacja: {relation.PARENT_ID} -> {relation.CHILD_ID}")
``` 

### Praca z grafami
``` python
import networkx as nx

# Po konwersji do grafu NetworkX
synset_graph = mapper.convert_to_synset_graph()

# Analiza właściwości grafu
print(f"Liczba węzłów: {synset_graph.number_of_nodes()}") 
print(f"Liczba krawędzi: {synset_graph.number_of_edges()}") 
print(f"Czy graf jest spójny: {nx.is_connected(synset_graph.to_undirected())}")

# Znajdowanie najkrótszych ścieżek
if nx.is_connected(synset_graph.to_undirected()): 
    path = nx.shortest_path(synset_graph, source=node1, target=node2) 
    print(f"Najkrótsza ścieżka: {path}")
``` 

---

## Rozwiązywanie problemów

### Problemy z połączeniem do bazy danych

1. **Sprawdź konfigurację** - upewnij się, że plik konfiguracji zawiera poprawne dane
2. **Sprawdź dostępność serwera** - zweryfikuj czy serwer MySQL jest uruchomiony
3. **Sprawdź uprawnienia** - użytkownik musi mieć odpowiednie uprawnienia do bazy

### Problemy z pamięcią podczas konwersji

1. **Użyj parametru `--limit`** - ogranicz liczbę przetwarzanych rekordów
2. **Zwiększ dostępną pamięć** - dla dużych zbiorów danych
3. **Przetwarzaj fragmentami** - użyj mniejszych partii danych

### Problemy z pobieraniem Wikipedii

1. **Sprawdź połączenie internetowe**
2. **Użyj mniejszego limitu** - Wikipedia może ograniczać liczbę żądań
3. **Sprawdź logi** - włącz DEBUG level dla szczegółowych informacji

---

## Rozwój projektu

### Status projektu
Projekt jest aktywnie rozwijany. Aktualna wersja: **1.0.0**

### Współpraca
Zapraszamy do współpracy! Jeśli chcesz wnieść swój wkład:

1. **Fork** repozytorium
2. **Utwórz branch** na swoją funkcjonalność
3. **Wprowadź zmiany** z testami
4. **Wyślij Pull Request**

### Zgłaszanie błędów
Błędy i propozycje funkcjonalności można zgłaszać przez GitHub Issues.

### Testowanie
Projekt używa modułu `unittest`. Uruchomienie testów:
```
bash python -m unittest discover -s tests -v
``` 

---

## Dokumentacja API

Szczegółowa dokumentacja każdej klasy i metody znajduje się w docstringach w kodzie źródłowym.
Każda funkcja zawiera opis parametrów, wartości zwracanych oraz przykłady użycia.

### Główne klasy API:
- `PlWordnetAPIBase` - abstrakcyjny interfejs API
- `PlWordnetAPIMySQLDbConnector` - implementacja dla MySQL
- `PolishWordnet` - główna klasa do pracy z danymi
- `GraphMapper` - konwersja do grafów NetworkX
- `CommentParser` - parsowanie komentarzy PLWordNet

---

## Licencja

Apache 2.0 License - szczegóły w pliku [LICENSE](LICENSE).

---

## Kontakt

Projekt rozwijany przez RadLab.Dev Team.
Repository: https://github.com/radlab-dev-group/radlab-plwordnet

