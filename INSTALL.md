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

```text
plwordnet-milvus \                                                                                                                                                                                                                                                                   
  --milvus-config=resources/configs/milvus-config-pk.json \                                                                                                                                                                                                                          
  --embedder-config=resources/configs/embedder-config.json \                                                                                                                                                                                                                         
  --nx-graph-dir="resources/plwordnet_4_5/full/graphs/full/nx/graphs/" \                                                                                                                                                                                                             
  --relgat-mapping-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm" \                                                                                                                                                                            
  --relgat-dataset-directory="resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way" \                                                                                                                                                        
  --log-level=DEBUG \                                                                                                                                                                                                                                                                
  --export-relgat-dataset                                                                                                                                                                                                                                                            
2025-10-21 18:13:49,705 - plwordnet_ml.embedder.model_config - INFO - model_config.py:224- Loaded active model configuration: 'o78zalgm' - Semantic-v0.3-o78zalgm                                                                                                                    
2025-10-21 18:13:49,706 - apps.cli.plwordnet_milvus_cli - INFO - plwordnet_milvus_cli.py:27- Starting plwordnet-milvus                                                                                                                                                               
2025-10-21 18:13:49,706 - apps.cli.plwordnet_milvus_cli - DEBUG - plwordnet_milvus_cli.py:28- Arguments: {'db_config': 'resources/plwordnet-mysql-db.json', 'use_database': False, 'nx_graph_dir': 'resources/plwordnet_4_5/full/graphs/full/nx/graphs/', 'log_level': 'DEBUG', 'limi
t': None, 'milvus_config': 'resources/configs/milvus-config-pk.json', 'embedder_config': 'resources/configs/embedder-config.json', 'prepare_database': False, 'prepare_base_embeddings_lu': False, 'prepare_mean_empty_base_embeddings_lu': False, 'prepare_base_embeddings_synset': 
False, 'export_relgat_mapping': False, 'export_relgat_dataset': True, 'relgat_mapping_directory': 'resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm', 'relgat_dataset_directory': 'resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/d
ataset_syn_two_way', 'device': 'cpu'}                                                                                                                                                                                                                                                
2025-10-21 18:13:49,706 - plwordnet_handler.base.connectors.nx.nx_loader - INFO - nx_loader.py:32- Loading data from NetworkX graphs
2025-10-21 18:13:49,706 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:416- Loading graph from resources/plwordnet_4_5/full/graphs/full/nx/graphs/lexical_units.pickle                                                               18:53:23 [13/224]
2025-10-21 18:13:55,765 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:425- Loaded graph with 514016 nodes and 399968 edges                                                                                                                           
2025-10-21 18:13:55,766 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:416- Loading graph from resources/plwordnet_4_5/full/graphs/full/nx/graphs/synsets.pickle                                                                                      
2025-10-21 18:14:02,896 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:425- Loaded graph with 354316 nodes and 1491277 edges                                                                                                                          
2025-10-21 18:14:02,896 - plwordnet_handler.base.connectors.nx.nx_connector - INFO - nx_connector.py:73- Successfully loaded NetworkX graphs from resources/plwordnet_4_5/full/graphs/full/nx/graphs                                                                                 
2025-10-21 18:14:02,896 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:512- Loading relation_types from resources/plwordnet_4_5/full/graphs/full/nx/graphs/relation_types.pickle                                                                      
2025-10-21 18:14:07,852 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:522- Loaded 306 relation_types from resources/plwordnet_4_5/full/graphs/full/nx/graphs/relation_types.pickle                                                                   
2025-10-21 18:14:07,852 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:81- Successfully loaded relation_types!                                                                                                                                        
2025-10-21 18:14:07,852 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:512- Loading unit_and_synsets from resources/plwordnet_4_5/full/graphs/full/nx/graphs/lu_in_synsets.pickle                                                                     
2025-10-21 18:14:08,572 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:522- Loaded 514018 unit_and_synsets from resources/plwordnet_4_5/full/graphs/full/nx/graphs/lu_in_synsets.pickle                                                               
2025-10-21 18:14:08,573 - plwordnet_handler.base.connectors.nx.nx_connector - DEBUG - nx_connector.py:87- Successfully loaded lexical_units_in_synsets!                                                                                                                              
2025-10-21 18:14:08,573 - plwordnet_handler.base.connectors.nx.nx_connector - INFO - nx_connector.py:89- Successfully loaded NetworkX graphs resources from resources/plwordnet_4_5/full/graphs/full/nx/graphs                                                                       
2025-10-21 18:14:08,573 - plwordnet_ml.cli.wrappers - DEBUG - base_wrapper.py:171- Polish Wordnet connection established                                                                                                                                                             
2025-10-21 18:14:08,573 - plwordnet_ml.cli.wrappers - INFO - wrappers.py:396- Exporting RELGat dataset to directory resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm                                                                                         
2025-10-21 18:14:08,844 - synset-embeddings.log - INFO - base_connector.py:95- Connected to Milvus at 192.168.100.67:19533
2025-10-21 18:14:08,844 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - DEBUG - aligned_dataset_id.py:119- Loading RelGAT mapping from resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm
2025-10-21 18:14:08,844 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - INFO - aligned_dataset_id.py:315- Loading mappings from path resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm
2025-10-21 18:14:08,847 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - DEBUG - aligned_dataset_id.py:324- Loading mapping rel_align_to_original from resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/rel_align_to_original.json
2025-10-21 18:14:08,851 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - DEBUG - aligned_dataset_id.py:324- Loading mapping rel_original_to_align from resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/rel_original_to_align.json
2025-10-21 18:14:08,854 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - DEBUG - aligned_dataset_id.py:324- Loading mapping rel_name_to_aligned_id from resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/rel_name_to_aligned_id.json
2025-10-21 18:14:08,857 - plwordnet_ml.dataset.aligned_id.aligned_dataset_id - DEBUG - aligned_dataset_id.py:324- Loading mapping aligned_id_to_rel_name from resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/aligned_id_to_rel_name.json
2025-10-21 18:14:08,861 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:105- Exporting RelGAT mappings
2025-10-21 18:14:08,862 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:127- Preparing data to export
2025-10-21 18:14:08,862 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:228-  - preparing embeddings for each lexical unit
2025-10-21 18:14:17,926 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:237-   -> number of lexical units: 295419
Retrieving embeddings from Milvus: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 295419/295419 [23:34<00:00, 208.85it/s]
2025-10-21 18:37:52,421 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:182-  - preparing relation triplets (src_idx, dst_idx, rel_name)
2025-10-21 18:37:53,985 - plwordnet_handler.dataset.exporter.relgat - WARNING - relgat.py:354- Cannot find mapping for relation 144. Skipping.
2025-10-21 18:37:54,026 - plwordnet_handler.dataset.exporter.relgat - WARNING - relgat.py:354- Cannot find mapping for relation 143. Skipping.
2025-10-21 18:37:54,032 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:189-    - relations with both embeddings : 180962
2025-10-21 18:37:55,573 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:200-    - synonyms with both embeddings : 178400
2025-10-21 18:37:55,601 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:210-    - found different relations: 56
2025-10-21 18:37:55,605 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:213-    - all relations: 359362
2025-10-21 18:37:55,607 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:149- Storing RelGAT mappings to resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way
2025-10-21 18:37:55,607 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:158-  - exporting resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way/lexical_units_embedding.pickle
2025-10-21 18:38:37,347 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:158-  - exporting resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way/relation_to_idx.json
2025-10-21 18:38:37,397 - plwordnet_handler.dataset.exporter.relgat - INFO - relgat.py:158-  - exporting resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way/relations_triplets.json
2025-10-21 18:38:39,223 - plwordnet_ml.cli.wrappers - INFO - wrappers.py:426- Successfully exported to resources/plwordnet_4_5/full/relgat/aligned-dataset-identifiers/o78zalgm/dataset_syn_two_way
```