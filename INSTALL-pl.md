## INSTALL – Kompletny przewodnik budowy zbiorów i embeddera dla **Słowosieci**

---

### Spis treści

1. [Wymagania wstępne](#wymagania-wstępne)
2. [Instalacja bazy Słowosieci](#instalacja-bazy-słowosieci)
3. [Krok 1 – Przygotowanie wag relacji](#krok‑1‑przygotowanie-wag-relacji)
4. [Krok 2 – Budowa grafu artykułów Wikipedii](#krok‑2‑budowa-grafu-artykulów-wikipedii)
5. [Krok 3 – Tworzenie zbioru danych embeddera](#krok‑3‑tworzenie-zbioru-danych-embeddera)
6. [Krok 4 – Trening embeddera (bi‑encodera)](#krok‑4‑trening-embeddera-biencodera)
7. [Krok 5 – Generowanie embeddingów dla znaczeń](#krok‑5‑generowanie-embeddingów-dla-znaczeń)
8. [Krok 6 – Eksport danych dla RelGAT](#krok‑6‑eksport-danych-dla-relgat)
9. [Skrócone ścieżki (gotowe artefakty)](#skrót‑ścieżki‑gotowe‑artefakty)

---  

## Wymagania wstępne

| Element                | Minimalna wersja | Uwagi                                                                                 |
|------------------------|------------------|---------------------------------------------------------------------------------------|
| **Python**             | 3.10.6           | Używany wirtualny środowisko `virtualenv`                                             |
| **MySQL**              | 5.7+             | Do przechowywania bazy Słowosieci                                                     |
| **Milvus**             | 2.2+             | Baza wektorowa                                                                        |
| **CUDA** (opcjonalnie) | 11+              | Przyspieszenie treningu i inferencji                                                  |
| **Pakiety Python**     | –                | Zainstalowane z `requirements.txt` (np. `requests`, `pandas`, `networkx`, `torch`, …) |

Instalację pakietów wykonujemy w aktywowanym środowisku:

```shell script
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---  

## Instalacja bazy Słowosieci

1. Pobierz najnowszy dump **Słowosieci** z oficjalnej strony:  
   <http://plwordnet.pwr.wroc.pl/wordnet/download>
2. Utwórz bazę MySQL i załaduj dump:

```shell script
# 1️⃣ Połącz się z serwerem MySQL
mysql -u root -p

# 2️⃣ Utwórz nową bazę
CREATE DATABASE wordnet_work;

# 3️⃣ Załaduj plik dumpu (przykładowa nazwa)
mysql -u USER -p wordnet_work < wordnet_work_4_5.sql
```

> **Uwaga:** Nazwy użytkownika oraz hasła dopasuj do własnej konfiguracji.

---  

## Krok 1 – Przygotowanie wag relacji

Relacje pomiędzy synsetami muszą mieć przypisane wagi, które są wykorzystywane przy budowie grafu oraz przy trenowaniu
embeddera.

* **Opcja A – użycie gotowego pliku**  
  Repozytorium zawiera aktualny plik `resources/mappings/relation-types-weights-hist.xlsx`. Wystarczy go pozostawić w
  miejscu domyślnym.

* **Opcja B – samodzielne przygotowanie**
    1. Utwórz arkusz XLSX ze wszystkimi typami relacji.
    2. W kolumnie `embedder_weight_coarse` wpisz wagi (liczby rzeczywiste).
    3. Uruchom skrypt, który przetworzy arkusz:

```shell script
bash scripts/0-plwordnet-cli-prepare-relations.sh
```

> **Uwaga:** Skrypt można dostosować, edytując zmienne wejściowe.

---  

## Krok 2 – Budowa grafu artykułów Wikipedii

Graf łączy jednostki leksykalne (LU) i synsety z tekstami Wikipedii.

* **Pełna budowa od zera** (zalecane, jeśli chcesz własne dane):

```shell script
bash scripts/1-plwordnet-cli-dump-to-nx.sh
```

> Skrypt pobiera dump Wikipedii, przetwarza go na graf `networkx` i zapisuje w katalogu
`resources/plwordnet_4_5/full/graphs/...`.  
> W trakcie przetwarzania wykorzystywany jest lokalny serwis **OpenAPI** do korekcji interpunkcji – przy dużej liczbie
> wątków może to być czasochłonne.

* **Alternatywa – użycie gotowego grafu**  
  Jeśli nie chcesz budować grafu samodzielnie, zainstaluj pełne zależności aplikacji (`FULL/TEST_GRAPH`) i pomiń ten
  krok.

* **Rozwiązywanie problemów**  
  Po zakończeniu budowy może pojawić się komunikat `... node has no data ...`. W takim wypadku uruchom ponownie skrypt –
  pamięć podręczna (`cache`) zapewnia, że artykuły nie będą pobierane ponownie.

---  

## Krok 3 – Tworzenie zbioru danych embeddera

### 3.1 Zrzut surowych relacji

```shell script
bash scripts/2-plwordnet-cli-dump-embedder-raw.sh
```

* Skrypt generuje plik `.../embedder/plwn_4_5_embedder_raw.jsonl` zawierający pary zdań `{zdanie_1, zdanie_2, rel_i}`.
* Wykorzystuje plik wag relacji (z kroku 1).
* Parametr `--embedder-low-high-ratio 2.0` oznacza **2 negatywne przykłady na każdy pozytywny**.

### 3.2 Konwersja do finalnego formatu

```shell script
bash scripts/3-raw-embedder-to-proper-dataset.sh
```

* Tworzy podzielony na `train`/`test` zestaw w formacie `jsonl`.
* Domyślny podział: 90 % trening, 10 % test (`--train-ratio=0.90`).
* Dzieli dane na zdania (`--split-to-sentences`), uruchamia 32 wątki (`--n-workers=32`) i przetwarza w partiach po 500
  rekordów (`--batch-size=500`).

### 3.3 Deduplication (usuwanie duplikatów)

```shell script
bash scripts/4-deduplicate-embedder-dataset.sh
```

* Usuwa powtarzające się rekordy, filtruje przykłady krótsze niż 25 znaków oraz zapisuje czysty zbiór gotowy do
  treningu.

> Po wykonaniu powyższych trzech pod‑skryptów otrzymujesz kompletny, zbalansowany i oczyszczony dataset do wytrenowania
> embeddera.

---  

## Krok 4 – Trening embeddera (bi‑encodera)

Trening odbywa się na przygotowanym zbiorze przy użyciu modeli **EuroBERT**.

### Dostępne skrypty treningowe

| Skrypt                                  | Model         | Liczba parametrów |
|-----------------------------------------|---------------|-------------------|
| `run_train_biencoder_eurobert_0.61b.sh` | EuroBERT‑610M | 610 M             |
| `run_train_biencoder_eurobert_2.1b.sh`  | EuroBERT‑2.1B | 2.1 B             |

Przykład uruchomienia (model 610 M):

```shell script
bash plwordnet_ml/training_scripts/run_train_biencoder_eurobert_0.61b.sh
```

> **Uwaga:** Trening od zera wymaga kilku dni na typowym GPU. Zdecydowanie szybciej jest pobrać gotowe wagi z
> HuggingFace:  
> <https://huggingface.co/radlab/semantic-euro-bert-encoder-v1>

---  

## Krok 5 – Generowanie embeddingów dla znaczeń

Po uzyskaniu wytrenowanego (lub pobranego) modelu tworzymy wektorowe reprezentacje dla wszystkich LU i synsetów oraz
zapisujemy je w bazie **Milvus**.

### 5.1 Inicjalizacja bazy Milvus

```shell script
plwordnet-milvus \
  --log-level=DEBUG \
  --milvus-config=resources/configs/milvus-config.json \
  --prepare-database
```

### 5.2 Przygotowanie embeddingów (real + fake)

```shell script
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

* **`--prepare-base-embeddings-lu`** – wylicza embeddingi dla jednostek leksykalnych.
* **`--prepare-base-embeddings-synset`** – wylicza embeddingi dla synsetów.
* **`--prepare-base-mean-empty-embeddings-lu`** – tworzy tzw. *fake* embeddingi (średnie wektory) wykorzystywane przy
  brakujących danych.

Po uruchomieniu zobaczysz log podobny do:

```
2025-10-19 15:42:10,922 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:73- Connected to default Milvus database at 192.168.100.67:19533
...
2025-10-19 15:42:21,612 - plwordnet_handler.base.connectors.milvus.initializer - INFO - initializer.py:162- Milvus WordNet handler initialized successfully
```

---  

## Krok 6 – Eksport danych dla RelGAT

Aby wytrenować model RelGAT potrzebny jest specjalny zestaw danych (mapping + same przykłady).

```shell script
plwordnet-milvus \
  --milvus-config=resources/configs/milvus-config-pk.json \
  --nx-graph-dir="resources/plwordnet_4_5/full/graphs/full/nx/graphs/" \
  --relgat-mapping-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm" \
  --relgat-dataset-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way" \
  --log-level=DEBUG \
  --export-relgat-dataset \
  --export-relgat-mapping
```

* **`--export-relgat-dataset`** – zapisuje gotowy zestaw treningowy w formacie wymaganym przez RelGAT.
* **`--export-relgat-mapping`** – generuje plik mapujący identyfikatory w grafie na identyfikatory używane w modelu.

---  

## Skrót ścieżki – Gotowe artefakty

| Cel                                     | Skrypt / komenda                                                     | Co otrzymasz                                                             |
|-----------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| Gotowy graf z Wikipedii                 | `bash scripts/1-plwordnet-cli-dump-to-nx.sh` (lub pobranie gotowego) | `resources/plwordnet_4_5/full/graphs/...`                                |
| Gotowy plik wag relacji                 | `resources/mappings/relation-types-weights-hist.xlsx`                | Gotowy arkusz Excel                                                      |
| Gotowy dataset embeddera (train + test) | `bash scripts/4-deduplicate-embedder-dataset.sh`                     | `resources/plwordnet_4_5/full/embedder/plwn_4_5_embedder_dataset/*.json` |
| Wytrenowane modele                      | Pobranie z HuggingFace **lub** uruchomienie `run_train_biencoder_*`  | `OUT_DIR/.../biencoder/<timestamp>_...`                                  |
| Pełna inicjalizacja Milvus + embeddingi | `bash scripts/6-plwordnet-milvus-full-init.sh`                       | Baza Milvus gotowa do zapytań                                            |
| Eksport RelGAT                          | `bash scripts/7-plwordnet-milvus-relgat-export.sh`                   | `resources/.../relgat/...`                                               |

---

### Dodatkowe uwagi

* Wszystkie skrypty znajdują się w katalogu `scripts/`.
* Jeśli chcesz używać własnych konfiguracji (np. inny host Milvus, inny model BERT), edytuj odpowiednie pliki w
  `resources/configs/`.
* W razie problemów z zależnościami systemowymi (np. brak `libmysqlclient`), sprawdź sekcję **Issues** w repozytorium
  lub otwórz nowy ticket.

---  

> **Gotowe!** Po przejściu powyższych kroków masz kompletny ekosystem: baza danych, graf, zbiór treningowy, wytrenowany
> embedder oraz gotowe wektory w Milvus, gotowe do dalszych eksperymentów (np. wyszukiwanie semantyczne, klasyfikacja
> relacji, model RelGAT).

*Powodzenia!* 🚀  