import spacy
from typing import List

from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.similar_search_handler import (
    MilvusWordNetSemanticSearchHandler,
)
from plwordnet_trainer.embedder.model_config import BiEncoderModelConfig
from plwordnet_trainer.embedder.bi_encoder import BiEncoderEmbeddingGenerator


emb_generator = BiEncoderEmbeddingGenerator(
    model_config=BiEncoderModelConfig.from_json_file(
        config_path="resources/embedder-config.json"
    ),
    device="cpu",
    normalize_embeddings=True,
)

milvus = MilvusWordNetSemanticSearchHandler(
    config=MilvusConfig.from_json_file(
        config_path="resources/milvus-config-pk.json"
    ),
    auto_connect=True,
)

nlp = spacy.load("pl_core_news_lg")


def text_to_base_forms(text: str) -> List[str]:
    doc = nlp(text)
    base_forms = []
    for token in doc:
        if not token.is_punct and not token.is_space and len(token.text) > 1:
            base_forms.append(token.lemma_)
    return base_forms


class DataStrRepr:
    @staticmethod
    def lu_res_as_str(lu_dict):
        lemma = lu_dict["lemma"]
        pos = lu_dict["pos"]
        domain = lu_dict["domain"]
        variant = lu_dict["variant"]
        lu_id_res = lu_dict["lu_id"]
        return f"{lemma}.{variant} (d:{domain} p:{pos}) ID={lu_id_res}"

    @staticmethod
    def show_lu_results(results):
        for _idx, r in enumerate(results):
            dist = r["distance"]
            r_str = DataStrRepr.lu_res_as_str(r["entity"])
            print(f"\t{_idx + 1}. [{dist:.16f}] {r_str}")


class LuEmbeddingSearch:

    @staticmethod
    def run_lu_search():
        while True:
            try:
                lu_id = input(
                    "Podaj identyfikator lu_id (lub 'quit' aby zakończyć): "
                ).strip()
                if lu_id.lower() == "quit":
                    break

                if not lu_id:
                    print("Podaj prawidłowy identyfikator!")
                    continue
                else:
                    try:
                        lu_id = int(lu_id)
                    except ValueError:
                        print("lu_id musi być liczbą!")
                        continue

                result = milvus.get_lexical_unit_embedding(lu_id=lu_id)
                if result is not None:
                    most_sim = milvus.search_most_similar_lu(
                        query_embedding=result["embedding"],
                        top_k=5,
                        metric_type="COSINE",
                    )

                    q_str = DataStrRepr.lu_res_as_str(result)
                    print(f" Most similar to: {q_str}")
                    DataStrRepr.show_lu_results(results=most_sim)
                    print("-" * 50)
                    print()
                else:
                    print(f"Nie znaleziono query_embedding dla lu_id={lu_id}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Błąd podczas wyszukiwania: {e}")
                continue


class TextSearch:
    @staticmethod
    def run_text_search():
        while True:
            try:
                text_to_disamb = input(
                    "Podaj tekst do ujednoznacznienia (quit - aby wyjść): "
                ).strip()
                if text_to_disamb.lower() == "quit":
                    break

                if not text_to_disamb or len(text_to_disamb) < 4:
                    print("Podaj tekst (co najmniej 3 znaki)")
                    continue

                query_embedding = emb_generator.text_to_embedding(
                    text=text_to_disamb
                )

                text_to_disamb_base = text_to_base_forms(text=text_to_disamb)
                print(" BASE FORMS [", text_to_disamb_base, "]")

                #
                # print("query_embedding=", query_embedding)
                if query_embedding is not None:
                    most_sim = milvus.search_most_similar_lu(
                        query_embedding=query_embedding,
                        top_k=len(text_to_disamb.split()) * 3,
                        metric_type="COSINE",
                    )

                    print(f" Most similar lu")
                    print("-" * 50)
                    DataStrRepr.show_lu_results(results=most_sim)
                    print("-" * 50)
                    print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Błąd podczas wyszukiwania: {e}")
                continue


while 1:
    try:
        opt_menu = input("Podaj komendę: ")
        if opt_menu is None or not len(opt_menu.strip()):
            continue

        if opt_menu.lower() == "quit":
            break
        elif opt_menu.lower() == "lu_search":
            LuEmbeddingSearch.run_lu_search()
        elif opt_menu.lower() == "text_search":
            TextSearch.run_text_search()
        else:
            print("")
            print("Dostępne komendy: ")
            print("  quit        - wyjście")
            print("  lu_search   - wyszukiwanie jednostek")
            print("  text_search - ujednoznacznianie")
            print("")
    except KeyboardInterrupt:
        break

# from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
# from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database
# connector = connect_to_mysql_database(
#     db_config_path="resources/plwordnet-mysql-db.json",
#     connect=True,
# )
# pl_wn = PolishWordnet(
#     connector=connector,
#     db_config_path=None,
#     nx_graph_dir=None,
#     extract_wiki_articles=False,
#     use_memory_cache=True,
#     show_progress_bar=False,
# )
