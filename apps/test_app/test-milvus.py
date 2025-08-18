from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database
from plwordnet_handler.base.connectors.milvus.similar_search_handler import (
    MilvusWordNetSemanticSearchHandler,
)


milvus_config = MilvusConfig.from_json_file(
    config_path="resources/milvus-config-pk-l2.json"
)

connector = connect_to_mysql_database(
    db_config_path="resources/plwordnet-mysql-db.json",
    connect=True,
)

pl_wn = PolishWordnet(
    connector=connector,
    db_config_path=None,
    nx_graph_dir=None,
    extract_wiki_articles=False,
    use_memory_cache=True,
    show_progress_bar=False,
)

milvus = MilvusWordNetSemanticSearchHandler(config=milvus_config, auto_connect=True)


def lu_res_as_str(lu_dict):
    lemma = lu_dict["lemma"]
    pos = lu_dict["pos"]
    domain = lu_dict["domain"]
    variant = lu_dict["variant"]
    lu_id_res = lu_dict["lu_id"]
    return f"{lemma}.{variant} (d:{domain} p:{pos}) ID={lu_id_res}"


def run_lu_search():
    while True:
        try:
            lu_id = input(
                "Podaj identyfikator lu_id (lub 'quit' aby zakończyć): "
            ).strip()
            if lu_id.lower() == "quit":
                print("Zakończenie programu.")
                break

            if not lu_id:
                print("Podaj prawidłowy identyfikator!")
                continue

            try:
                lu_id = int(lu_id)
            except ValueError:
                print("lu_id musi być liczbą!")
                continue

            result = milvus.get_lexical_unit_embedding(lu_id=lu_id)
            if result is not None:
                most_sim = milvus.search_most_similar_lu(
                    query_embedding=result["embedding"], top_k=5, metric_type="L2"
                )

                q_str = lu_res_as_str(result)
                print(f" Most similar to {q_str}")

                for _idx, r in enumerate(most_sim):
                    dist = r["distance"]
                    r_str = lu_res_as_str(r["entity"])
                    print(f"\t{_idx + 1}. [{dist}] {r_str}")
                print()

            else:
                print(f"Nie znaleziono query_embedding dla lu_id={lu_id}")

        except KeyboardInterrupt:
            print("\nPrzerwano przez użytkownika.")
            break
        except Exception as e:
            raise e
            print(f"Błąd podczas wyszukiwania: {e}")
            continue

    print("Program zakończony.")


run_lu_search()
