## Przygotowanie zbioru embeddera

Poniżej znajduje się opis przygotowania zbioru danych do uczenia embeddera semantycznego.
Zakłada się, że użytkownik dysponuje lokalnie grafami, lub połaczeniem do bazy.

Wcześniej należy posiadać również zainstalowaną bibliotekę `radlab-plwordnet`.
W katalogu głównum np. `pip3 install . --break && rm -rf build *.egg-info`

---

**Konwersja Słowosieci do podsawowego zbioru ebeddera**

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

---

**Przygotowanie danych do deduplikacji**

Po tym należy przekonwetować zbiór do poprawnego formatu z podziałem na dane treningowe i testowe.
Do tego należy wykorzystać aplikację `convert-raw-embedder-dump-to-dataset.py` z `apps/utils/embedder`.
W zależności, czy chcemy dzieć dodatkowo teksty z komentarzy, definicji, przykładów, emocji, Wikipedii
na zdania, należy podać flagę `--split-to-sentences`. W przypadku korzystania z dodakowego podziału
na zdania, mocno zaleca się wykorzystanie wielowątkowości (przełącznik `--n-workers` z opcjonalnie
ustawionym `--batch-size`). Dla przykładu (z `--train-ratio=0.93` -- `93%` trening, `7%` walidacja):

```bash
python3 plwordnet_trainer/embedder/apps/convert-plwn-dump-to-dataset.py \
	--jsonl-path=resources/emb_dataset/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl \
	--output-dir=resources/emb_dataset/embedding-dump-ratio-1.2-w-synonymy/embedder_nosent_train-0.93 \
	--train-ratio=0.93
```

lub z właczaonym podziałem na zdania i wielowątkowością (czas konwersji ~3 godziny, bez wielowątkości 30-50 godzin):

```bash
python3 plwordnet_trainer/embedder/apps/convert-plwn-dump-to-dataset.py \
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

---

**Deduplikacja danych treningowo-testowych**

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
