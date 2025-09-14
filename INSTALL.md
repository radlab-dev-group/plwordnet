## Instalacja bazy Słowosici i definicji dla EWN (English Wordnet)

Pobieramy Słowosieć z linku: http://plwordnet.pwr.wroc.pl/wordnet/download

```bash
mysql -u root -p

# Konfiguracja bazy, użytkownik etc...

>> CREATE DATABASE wordnet_work;

mysql -u USER -p wordnet_work < wordnet_work_4_5.sql
```

Pobieramy definicje dla PrincetonWordnet/[English Wordnet](https://github.com/globalwordnet/english-wordnet)

```bash
wget https://en-word.net/static/english-wordnet-2024.xml.gz

gunzip english-wordnet-2024.xml.gz
```


