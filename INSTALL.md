# Proces budowy zbiorów i embeddera

---

## Instalacja bazy Słowosici

Pobieramy Słowosieć z linku: http://plwordnet.pwr.wroc.pl/wordnet/download

```bash
mysql -u root -p

# Konfiguracja bazy, użytkownik etc...

>> CREATE DATABASE wordnet_work;

mysql -u USER -p wordnet_work < wordnet_work_4_5.sql
```

---

## Tworzenie podstawowych struktur

### 1. Przygotowanie pliku z wagami relacji

Pierwszym krokiem jest przygotowanie pliku XLSX z wykazem relacji.  Relacje potrzebne są w procesie 
budowy zbioru embeddera (i konwersji do grafu). Dla każdej relacji przypisana powinna być waga.

**UWAGA!** Ten krok można pominąć, na repozytorium znajduje się plik z aktualnym wykazem relacji
i zaproponowanymi wagami plik:  
[relation-types-weights-hist.xlsx](resources/mappings/relation-types-weights-hist.xlsx).

**UWAGA!** Po utworzeniu samemu pliku z relacjami należy ustawić ich wagi. Waga z kolumny 
`embedder_weight_coarse` jest brana pod uwagę podczas budowania grafu i zbiorów embeddera.

Aby jednak samemu przygotować od początku plik z relacjami, można posłużyć się skryptem:

```bash 
bash scripts/0-plwordnet-cli-prepare-relations.sh
```
Ewentualnie modyfikując go do własnych potrzebb.

---

### 2. Przygotowanie grafu z artykułami z Wikipedii

**UWAGA!** ten krok można pominąć instalując zależności aplikacji (`FULL/TEST_GRAPH`).

W przypadku tworzenia grafu od początku samemu warto skorzystać z gotowego skryptu:

```bash
bash scripts/1-plwordnet-cli-dump-to-nx.sh
```

**UWAGA!** w skrypcie wykorzystywana jest lokalne `OpenAPI` do korekcji tekstów (poprawiona interpunkcja).
Przy małej liczbie wątków poprawiających proces ten może być bardzo czasochłonny, dlatego zalecamy
pobranie przygotowanego już grafu z wyekstrahwowamymi i poprawionymi już artykułami.

Jeżeli po budowie pokaże się komunikat typu: `... node hass no data ...`
należy uruchomić jeszcze raz skrypt (aktualnie, cachowanie zawsze jest uruchomione, dlatego zarówno
artykuły z Wikipedii jak i ich wersje czyste są już przygotowane)

---

### 3. Przygotowanie zbioru danych embeddera

Pierwszy krok to zrzut dostępnych relacji z "sąsiadującymi" definicjami, komentarzami,
anotacjami, emocjami itp. dla jednostek leksykalnych oraz synsetów. Ten zbiór zawierał będzie
pary `{zdanie_1, zdanie_2}` z relacji `rel_i`. Do przygotowania **podstawowego zbioru** 
można wykorzystać skrypt:
```bash
bash scripts/2-plwordnet-cli-dump-embedder-raw.sh
```

To co **ważne**, w skrypcie podana jest ścieżka do pliku z relacjami. Jeżeli pominąłeś krok 
 wag relacji, możesz skopiować ten, dostępny na repozytorium 
`resources/mappings/relation-types-weights-hist.xlsx`.

Podczas tworzenia zbioru podstawowego określa się stosunek przykładów negatywnych (czyli tych bez relacji)
do pozytywnych (z relacją). Wartość tę ustaswia się za pomocą argumentu `--embedder-low-high-ratio`
(w skryocie ustawione na `2.0` - czyli dwa przykłady negatywne na każdy pozytywny).

Po wykonaniu tego kroku, w pliku `--dump-embedder-dataset-to-file` zapisany zostanie podstawowy
zbiór danych (w formacie `jsonl`), który należy zdeduplikować, oczyścić z ewentualnych "śmieci" 
i przekonwertować na format do uczenia embeddera. Do tego celu można wykorzystać skrypt:

```bash
bash scripts/3-raw-embedder-to-proper-dataset.sh
```

Róznica między plikami wyjściowymi z `2-plwordnet-cli-dump-embedder-raw.sh`
a `3-raw-embedder-to-proper-dataset.sh` to przeznaczenie. Wyjście `2-...` to ogólne dane,
zaś wyjście z `3-raw-embedder-to-proper-dataset.sh` to gotowy zbiór danych, 
z podziałem na dane testowe i ewaluacyjne z podziałem `--train-ratio=0.90`.
Domyślnie zbiór dzielony jest na zdania (`--split-to-sentences`), które realizowane jest
przez `--n-workers=32` workerów w batchach `--batch-size=500`. Do **deduplikacji** danych,
można wykorzystać skrypt

```bash
bash scripts/4-deduplicate-embedder-dataset.sh
```
po tym procesie, posiadamy gotowy zbiór danych do wyuczenia embeddera. Zbiór:
 - posiada relacje międzyjęzykowe
 - przykłady negatywne
 - podzielony jest na zdania
 - zdeduplikowany na całym zbiorze
 - wybierze tylko takie przykłady, w których oba teksty mają co najmniej 25 znaków

---

### 4. Uczenie embeddera

Posiadając już przygotowany zbiór danych z wcześniejszych kroków można przystąpić
do trenowania modelu `bi-encodera`. W naszym przypadku jako podstawowy model
wykorzystaliśmy modele `EuroBERT/EuroBERT-610m` oraz `EuroBERT/EuroBERT-2.1B`. 

W katalogu `plwordnet_ml/training_scripts` znajdują się skrypty uczące:

 - `run_train_biencoder_eurobert_0.61b.sh` - uczenie modelu EuroBERT w architekturze 610m parametrów
 - `run_train_biencoder_eurobert_2.1b.sh` - EuroBERT w architekturze 2.1b parametrów

wykorzystują one kod trenera z modułu `plwordnet_ml/embedder/trainer/train-bi-encoder.py`.

Trenowanie modeli od początku jest dość czasochłonne, dlatego zalecamy pobranie
wytrenowanych wag modeli z naszego huggingface, aktualnie udostępniamy wagi dla modelu 610m parametrów:
[radlab/semantic-euro-bert-encoder-v1](https://huggingface.co/radlab/semantic-euro-bert-encoder-v1).

### 5. Przygotowanie embeddingów dla znaczeń

Po wytrenowaniu embeddera (lub wykorzystaniu już wytrenowanego -- zalecane)
można przystapić do tworzenia reprezentacji wektorowych dla znaczeń. 

**UWAGA** można pominąć kroki podane niżej, jeżeli wykorzysta się skrypt
```bash
bash scripts/6-plwordnet-milvus-full-init.sh
```


**Przygotowanie bazy semantycznej**

Na początku należy zainicjalizować bazę danych semantyczną, można to zrobić przy pomocy CLI `plwordnet-milvus`:

```bash
plwordnet-milvus 
    --log-level=DEBUG
    --milvus-config=resources/milvus-config.json 
    --prepare-database
```

powinien pojawić się log typu:


```text
2025-10-19 15:42:10,922 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:73- Connected to default Milvus database at 192.168.100.67:19533
2025-10-19 15:42:11,123 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:186- Created database: slowosiec_4_5_20250926_o78zalgm
2025-10-19 15:42:11,127 - plwordnet_handler.base.connectors.milvus.initializer - INFO - base_connector.py:95- Connected to Milvus at 192.168.100.67:19533
2025-10-19 15:42:12,259 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:297- Created collection: base_synset_embeddings
2025-10-19 15:42:13,611 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:318- Created collection: base_lu_embeddings
2025-10-19 15:42:14,502 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:340- Created collection: base_lu_examples_embeddings
2025-10-19 15:42:21,612 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:240- Created IVF_FLAT indexes on collections
2025-10-19 15:42:21,612 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:162- Milvus WordNet handler initialized successfully
2025-10-19 15:42:21,620 - plwordnet_handler.base.connectors.milvus.initializer - INFO - base_connector.py:95- Connected to Milvus at 192.168.100.67:19533
```

**Przygotowanie podstawowych (i fake) embeddingów**

```bash
# prepare database
plwordnet-milvus \
  --log-level=DEBUG \
  --milvus-config resources/configs/milvus-config.json \
  --prepare-database


# Base and fake embeddings
plwordnet-milvus \
  --milvus-config=resources/configs/milvus-config.json \
  --embedder-config=resources/configs/embedder-config.json \
  --nx-graph-dir="resources/plwordnet_4_5/full/graphs/full/nx/graphs/" \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-base-embeddings-lu \
  --prepare-base-embeddings-synset \
  --prepare-base-mean-empty-embeddings-lu
```


### 6. Przygotowanie datasetu do RelGAT trainera

```bash
plwordnet-milvus \
  --milvus-config=resources/configs/milvus-config-pk.json \
  --nx-graph-dir="resources/plwordnet_4_5/full/graphs/full/nx/graphs/" \
  --relgat-mapping-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm" \
  --relgat-dataset-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way" \
  --log-level=DEBUG \
  --export-relgat-dataset \
  --export-relgat-mapping
```