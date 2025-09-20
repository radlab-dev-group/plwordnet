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

### Przygotowanie pliku z wagami relacji

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

### Przygotowanie grafu z artykułami z Wikipedii

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

# cdn.
...