### Przykład połaczenia

**Bezpośrednio z parametrami**:
``` python
handler = MilvusWordNetSchemaHandler(
    host="localhost",
    port="19530",
    user="admin",
    password="password",
    db_name="wordnet"
)
```

**Wykorzystując `MilvusConfig`**:
``` python
config = MilvusConfig.from_json_file("milvus_config.json")
handler = MilvusWordNetSchemaHandler(config=config)
```

**Bezpośrednio z konfigu**:
``` python
handler = MilvusWordNetSchemaHandler.from_config_file("milvus_config.json")
```

**Przykłąd konfigu `milvus_config.json`**:
``` json
{
    "host": "localhost",
    "port": "19530",
    "user": "admin",
    "password": "password123",
    "db_name": "wordnet"
}
```
