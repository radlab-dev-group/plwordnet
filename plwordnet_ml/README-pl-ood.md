# Opis działania moduły `plwordnet_ml` oraz `plwordnet-milvus`

## Przygotowywanie zbiorów danych

### Przygotowanie zbioru embeddera

Poniżej znajduje się opis przygotowania zbioru danych do uczenia embeddera semantycznego.
Zakłada się, że użytkownik dysponuje lokalnie grafami, lub połaczeniem do bazy.

Wcześniej należy posiadać również zainstalowaną bibliotekę `radlab-plwordnet`.
W katalogu głównum np. `pip3 install . --break && rm -rf build *.egg-info`

#### Konwersja Słowosieci do podsawowego zbioru ebeddera

Na początku (korzystając z `plwordnet-cli`) należy wyeksportować (`--dump-embedder-dataset-to-file`) 
dowolnym  connectorem (`--use-database` lub `--nx-graphs-dir`) zawartośc bazy do podstawowego pliku 
embeddera, np za pomocą komenty (z radio _negatywnych_ do _pozytywnych_ `1.2`):

```bash
plwordnet-cli \
  --xlsx-relations-weights=resources/mappings/relation-types-weights-hist.xlsx \
  --dump-embedder-dataset-to-file=resources/emb_dataset/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl \
  --log-level=DEBUG \
  --nx-graph-dir=resources/slowosiec_full/nx/graphs/ \
  --embedder-low-high-ratio=1.2
```

#### Przygotowanie danych do deduplikacji

Po tym należy przekonwetować zbiór do poprawnego formatu z podziałem na dane treningowe i testowe.
Do tego należy wykorzystać aplikację `convert-raw-embedder-dump-to-dataset.py` z `apps/utils/embedder`.
W zależności, czy chcemy dzieć dodatkowo teksty z komentarzy, definicji, przykładów, emocji, Wikipedii
na zdania, należy podać flagę `--split-to-sentences`. W przypadku korzystania z dodakowego podziału
na zdania, mocno zaleca się wykorzystanie wielowątkowości (przełącznik `--n-workers` z opcjonalnie
ustawionym `--batch-size`). Dla przykładu (z `--train-ratio=0.93` -- `93%` trening, `7%` walidacja):

```bash
python3 plwordnet_ml/embedder/apps/convert-plwn-dump-to-dataset.py \
	--jsonl-path=resources/emb_dataset/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl \
	--output-dir=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_nosent_train-0.93 \
	--train-ratio=0.93
```

lub z właczaonym podziałem na zdania i wielowątkowością (czas konwersji ~3 godziny, 
bez wielowątkości 30-50 godzin):

```bash
python3 plwordnet_ml/embedder/apps/convert-plwn-dump-to-dataset.py \
	--jsonl-path=resources/emb_dataset/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl \
	--output-dir=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93 \
	--train-ratio=0.93 \
	--split-to-sentences \
	--n-workers=32 \
	--batch-size=50000


Converting samples...
Processing 12560515 samples in 252 batches using 32 workers
Converting sample batches: 100%|███████████████████████████████████████████████| 252/252 [2:56:55<00:00, 42.13s/it]
Split to train test
Writing dataset...
Done. Train: 13922579 samples, Test: 1047937 samples.
```

#### Deduplikacja danych treningowo-testowych

Kolejnym krokiem jest deduplikacja zbioru z wcześniejszego kroku.

```bash
python3 apps/utils/embedder/embedder-dataset-dedupliactor.py \
    --train=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/train.json 
    --test=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/test.json


Loading data...
Original number of examples in test file: 1047937
Original number of examples in train file: 13922579
------------------------------
Performing deduplication...
------------------------------
Deduplication finished.
Examples in test file after deduplication: 1027858 (removed 20079)
Examples in train file after deduplication: 12221037 (removed 1701542)
------------------------------
Data saved to file: resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/test_deduplicated.json
Data saved to file: resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93/train_deduplicated.json

Process complete!
```

## Semantyczna baza do przechowywania embeddingów

Podczas procesu tworzenia embeddingów semantycznych i innych, gdzie 
jako stały rezultat powstaje embedding, wykorzystywany jest Milvus
jako baza danych (zarówno składowanie jak i wyszukanie). 

Do obsługi połączenia z Milvusem i operacji na bazie Milvusa służy 
komenda konsolowa `plwordnet-milvus`. Pełna lista opcji dostępna po 
```bash
plwordnet-milvus --help
```

Aplikacja do łaczenia z bazą wykorzystuje plik konfiguracyjny `milvus-config.json` 
w postaci json (konfiguracja połączenia do bazy) podawany za pomocą przełącznika 
`--milvus-config`. Przykładowa zawartość pliku `milvus-config.json`:
```json
{
  "host": "localhost",
  "port": "19533",
  "user": "root",
  "password": "password123",
  "db_name": "wordnet_20250817"
}
```

Przed przystąpieniem do przygotowywania embedingów należy zainicjować
bazę, schematy, indeksy i kolekcje Milvusa za pomocą polecenia:
```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config-pk.json \
  --prepare-database
```

## Przygotowanie embeddingów

Przygotowanie embeddingów musi być poprzedzone procesem budowy modelu embeddera 
podstawowego. Proces opisany jest wcześniej. Po wyuczeniu embeddera należy podpiąć
model do CLI za pomocą pliku konfiguracyjnego `embedder-config.json`, w którym
podana jest konfiguracja modelu, który uzyty będzie jako **bazowy model** do uczenia
embeddingów semantycznych. Przykładowy plik konfiguracyjny:

```json
{
  "active": "b20igzw4_chk2300000",
  "models": {
    "b20igzw4_chk2300000": {
      "model_name": "Semantic-v0.2-b20igzw4_chk2300000",
      "wandb_url": "http://192.168.100.61:8080/pkedzia/plWordnet-semantic-embeddings/runs/b20igzw4",
      "model_path": "/mnt/data2/llms/models/radlab-open/embedders/plwn-semantic-embeddingss/v0.2/EuroBERT-610m/20250812_032338_embedder_sentsplit_train-0.93/checkpoint-2300000",
      "comment": "Model z synonimią, checkpoint w trakcie uczenia",
      "vec_size": 1152
    },
    "o7reqebo": {
      "model_name": "Semantic-v0.1-o7reqebo",
      "wandb_url": "http://192.168.100.61:8080/pkedzia/plWordnet-semantic-embeddings/runs/o7reqebo",
      "model_path": "/mnt/data2/llms/models/radlab-open/embedders/plwn-semantic-embeddingss/v0.1/EuroBERT-610m/biencoder/20250806_162431_full_dataset_ratio-2.0_train0.9_eval0.1/checkpoint-290268",
      "comment": "Model bez synonimii",
      "vec_size": 1152
    }
  }
}
```

Plik umożliwia definicję wielu modeli oraz wskazanie aktualnie aktywnego `active`.
Poszczególne pola to:
 - `models` zawiera zestaw definicji modeli do wykorzystania - słownik z mapowanie:
   - `nazwa kodowa modelu` -> `{definicja modelu}`, gdzie model definiowany jest za pomocą:
     - `model_name` - nazwa modelu wykorzystywana w aplikacjach (obowiązkowe)
     - `model_path` - scieżka do modelu (obowiązkowe)
     - `vec_size` - rozmiar embeddingu (**UWAGA** Milvus przy tworzeniu kolekcji sprawdza wymiar
 - `active` wskazuje na `nazwę kodową modelu`, który ładowany będzie jako domyślny
embeddingu, nie korzysta z tej wartości)
   - opcjonalnie `wandb_url` dla porządku: użyty model <-> eksperyment
   - opcjonalny komentarz w polu `comment`

Przykład wykorzystania modułu `BiEncoderModelConfigHandler` do obsługi konfigu embeddera 
z implementacją w `plwordnet_ml.embedder.model_config.BiEncoderModelConfig`:
```python
from plwordnet_ml.embedder.model_config import BiEncoderModelConfigHandler

# Initialize handler with configuration
handler = BiEncoderModelConfigHandler.from_json_file("embedder-config.json")

# Discover available models
models = handler.get_available_models()
print(f"Available models: {list(models.keys())}")

# Switch to a different model
if handler.set_active_model("alternative_model"):
    print(f"Switched to: {handler.model_name}")

# Query specific model configuration
model_config = handler.get_model_config("another_model")
if model_config:
    print(f"Vector size: {model_config.get('vec_size')}")
```

### Embeddingi podstawowe 

**Embeddingi podstawowe** budowane są na podstawie definicji jednostek leksykalnych.
W trakcie budowy wykorzystywany jest **model embeddera semantycznego**, którego
proces przygotowania przedstawiony jest wyżej.

W pierwszym kroku budowane są ebeddingi _podstawowe dla przykładów_ z jednostek
leksykalnych, a następnie bazując na _podstawowych embeddingach przykładów_ 
budowane są _embeddingi podstawowe dla jednostek_. Podczas budowy embeddingu
dla jednostki leksykalnej, brana jest pod uwagę strategia budowy (domyślnie `MEAN`).
Aby przygotować embeddinig podstawowe i zapisać je do Milvusa należy wykonać polecenie:
```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config.json \
  --embedder-config=resources/embedder-config.json \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-base-embeddings-lu \
  --nx-graph-dir=/path/to/plwordnet/graphs
```
**UWAGA!** Ważne aby embeddingi z modelu liczyć na karcie graficznej (w przykładzie
powyżej `--device="cuda:1"`), obliczenia na procesorze mogą być bardzo długie.

Kolejnym krokiem jest przygotowanie _fake embeddingów dla jednostek leksykalnych_.
Ponieważ wiele jednostek nie posiada przykładów/definicji, na podstawie których
można zbudować embedding podstawowy, należy uzupełnić dziury bazując na relacji
_synonimii_. W procesie powstają _fake embeddingi jednostek leksykalnych_, które
są w relacji _synosnimii_ z innymi, które posiadają co najmniej jeden
_podstawowy embedding jednostki leksykalnej_. Jeżeli jednostka nie ma w synsecie
innej jednostki z _podstawowym embeddingiem_ to w tym kroku zostanie pominięta.
Podczas tworzenia _fake embeddingów_ wykorzystywana jest strategia ich budowy,
która domyślnie ustawiona jest na `MEAN`. Aby przygotować _fake-emeddinig_
nalezy podać przełącznik `--prepare-base-mean-empty-embeddings-lu` 
do `plwordnet-milvus`. Przykład całej komendy:

```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config.json \
  --prepare-base-mean-empty-embeddings-lu \
  --nx-graph-dir=/path/to/plwordnet/graphs
```

Kolejnym krokiem jest przygotowanie podstawowych embeddingów dla synsetów z ważoną strategią budowy.
Synsety budowane są w oparciu o reprezentacje jednostek podstawowych (również fake).
Ważenie uwzględnia moc embeddingu jednostki leksykalnej wyrażonej w postaci liczby embeddingów
z przykładami użycia tej jednostki -- im więcej przykładów ma jednostka, 
tym mocniej uczestniczy w budowie. Dodatkowy współczynnik wygładzający
`embedder.generator.base_embeddings.SemanticEmbeddingGeneratorSynset.SMOOTH_FACTOR`
nie eliminuje przypadków, kiedy jednostka nie posiada przykładów, ale posiada embedding 
(jest to wspominana wcześniej **fake jednostka**) -- taka jednostka ma udział w propagacji 
na synset, jednak posiada najmniejszy wpływ. Aby dodać podstawowe embeddingi dla synsetów
należy wywołać `plwordnet-milvus` CLI z parametrem `--prepare-base-embeddings-synset`:

```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config.json \
  --prepare-base-embeddings-synset \
  --nx-graph-dir=/path/to/plwordnet/graphs
```

---

Oczywiście komendy można łączyć i wykonać jedno polecenie:
```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config-pk.json \
  --embedder-config=resources/embedder-config.json \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-database \
  --prepare-base-embeddings-lu \
  --prepare-base-mean-empty-embeddings-lu \
  --prepare-base-embeddings-synset \
  --nx-graph-dir=/path/to/plwordnet/graphs
```
czyli:
 - `--prepare-database` -- zainicjuje bazę danych (jeżeli nie ma)
 - `--prepare-base-embeddings-lu` -- przygotuje embeddinig podstawowe dla jednostek
 - `--insert-base-mean-empty-embeddings-lu` -- przygotuje podstawowe fake embeddingi dla jednostek
 - `--prepare-base-embeddings-synset` -- przygotowuje podstawowe embeddingi synsetów


### Model transformacji jednostek -- RelGAT

Model ten (oznaczmy go jako `RelGAT`) działa jako przekształcenie embeddingu `E1` (jednostki 1) 
do embeddngu `E2` (jednostki 2) przy pomocy relacji `R` zachodzącej między nimi. 
Model `RelGAT` to wyuczony model przekształceń w taki sposób aby :`E1-R->E2: RelGAT(E1) ~ E2`. 
Model ten zostanie wykorzystany jako macierz przekształceń w trakcie fuzji znaczeń, 
również w trakcie fuzji znaczeń niereprezentowanych za pomocą _base embeddingów_. 

Repozytorium opisujące model [relgat-lmm](https://github.com/radlab-dev-group/relgat-llm).

```bash
plwordnet-milvus \
  --milvus-config=resources/milvus-config-pk.json \
  --embedder-config=resources/embedder-config.json \
  --nx-graph-dir=/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs \
  --relgat-mapping-directory=resources/aligned-dataset-identifiers/ \
  --relgat-dataset-directory=resources/aligned-dataset-identifiers/dataset \
  --export-relgat-dataset \
  --export-relgat-mapping
```

wywołanie:
 - wyseksportuje mapowanie bazowe `--export-relgat-mapping` do katalogu
`resources/aligned-dataset-identifiers/`
 - przygotuje zbiór danych do uczenia modelu RelGAT (`--export-relgat-dataset`)
i wyeksportuje go do katalogu
`resources/aligned-dataset-identifiers/dataset`
 - do katalogu `--relgat-mapping-directory`