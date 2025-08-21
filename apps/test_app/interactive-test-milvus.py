import json
import spacy

from typing import List
from transformers import AutoTokenizer

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_ml.embedder.model_config import BiEncoderModelConfig
from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_ml.embedder.bi_encoder import BiEncoderEmbeddingGenerator
from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database
from plwordnet_handler.base.connectors.milvus.similar_search_handler import (
    MilvusWordNetSemanticSearchHandler,
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

bi_encoder_model_config = BiEncoderModelConfig.from_json_file(
    config_path="resources/embedder-config.json"
)

emb_generator = BiEncoderEmbeddingGenerator(
    model_config=bi_encoder_model_config,
    device="cpu",
    normalize_embeddings=True,
)

milvus = MilvusWordNetSemanticSearchHandler(
    config=MilvusConfig.from_json_file(
        config_path="resources/milvus-config-pk.json"
    ),
    auto_connect=True,
)

all_lexical_units = []
bi_encoder_tokenizer = None

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
        emb_type = lu_dict["type"]
        return f"{lemma}.{variant} (d:{domain} p:{pos}) ID={lu_id_res}:{emb_type}"

    @staticmethod
    def syn_res_as_str(syn_dict):
        syn_id = syn_dict["syn_id"]
        unitsstr = syn_dict["unitsstr"]
        emb_type = syn_dict["type"]
        return f"[{syn_id}:{emb_type}] {unitsstr}"

    @staticmethod
    def show_lu_results(results):
        for _idx, r in enumerate(results):
            dist = r["distance"]
            r_str = DataStrRepr.lu_res_as_str(r["entity"])
            print(f"\t{_idx + 1}. [{dist:.16f}] {r_str}")

    @staticmethod
    def show_syn_results(results):
        for _idx, r in enumerate(results):
            dist = r["distance"]
            r_str = DataStrRepr.syn_res_as_str(r["entity"])
            print(f"\t{_idx + 1}. [{dist:.16f}] {r_str}")


class LuEmbeddingSearch:

    @staticmethod
    def run_lu_search():
        while True:
            try:
                lu_id = input("Podaj identyfikator lu_id (q/b - powrót): ").strip()
                if lu_id.lower() in ["quit", "q", "back", "b"]:
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
    def run_text_search(use_synset: bool = False, use_lu: bool = False):
        if not any([use_synset, use_lu]):
            print("Należy wybrać: use_synset, use_lu ")
            return

        while True:
            try:
                text_to_disamb = input(
                    "Podaj tekst do ujednoznacznienia (q/b - powrót): "
                ).strip()
                if text_to_disamb.lower() in ["quit", "q", "back", "b"]:
                    break

                query_embedding = emb_generator.text_to_embedding(
                    text=text_to_disamb
                )

                text_to_disamb_base = text_to_base_forms(text=text_to_disamb)
                if not text_to_disamb or len(text_to_disamb) < 3:
                    print("Podaj tekst (co najmniej 3 znaki)")
                    continue

                print(" BASE FORMS", text_to_disamb_base)
                if query_embedding is not None:
                    filters = None
                    lemma_str = input("Podaj lemat [Enter aby pominąć]: ").strip()

                    print(f" Most similar")
                    print("-" * 50)

                    if use_synset:
                        if len(lemma_str):
                            filters = {"unitsstr": lemma_str}

                        most_sim = milvus.search_most_similar_synsets(
                            query_embedding=query_embedding,
                            top_k=len(text_to_disamb.split()) * 3,
                            metric_type="COSINE",
                            filters=filters,
                        )
                        DataStrRepr.show_syn_results(results=most_sim)
                    elif use_lu:
                        if len(lemma_str):
                            filters = {"lemma": lemma_str}

                        most_sim = milvus.search_most_similar_lu(
                            query_embedding=query_embedding,
                            top_k=len(text_to_disamb.split()) * 3,
                            metric_type="COSINE",
                            filters=filters,
                        )
                        DataStrRepr.show_lu_results(results=most_sim)

                    print("-" * 50)
                    print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Błąd podczas wyszukiwania: {e}")
                continue


class StatsModule:
    @staticmethod
    def historgram_lu_tokens_chars():
        global all_lexical_units
        global bi_encoder_tokenizer

        if bi_encoder_tokenizer is None:
            bi_encoder_tokenizer = AutoTokenizer.from_pretrained(
                bi_encoder_model_config.model_path
            )

        if not len(all_lexical_units):
            print("Pobieranie listy jednostek z konektora Słowosieci")
            all_lexical_units = pl_wn.get_lexical_units()

        print(
            "  * liczba jednostek bez filtrowania POS: ",
            len(all_lexical_units),
        )

        lemmas = set()
        for lu in all_lexical_units:
            lemmas.add(lu.lemma)
        print("  * liczba unikalnych lematów: ", len(lemmas))

        ch2len = {}
        tok2len = {}
        for lemma in lemmas:
            # Lemma as characters
            l_lemma = len(lemma.strip())
            if l_lemma not in ch2len:
                ch2len[l_lemma] = 0
            ch2len[l_lemma] += 1

            # Lemma as tokens
            tokens = bi_encoder_tokenizer.tokenize(lemma)
            l_tokens = len(tokens)
            if l_tokens not in tok2len:
                tok2len[l_tokens] = 0
            tok2len[l_tokens] += 1

        hist = {
            "characters": ch2len,
            "tokens": tok2len,
        }
        print(json.dumps(hist, indent=2, ensure_ascii=False))

        print("typ;dlugosc;czestosc")
        for h_type, histogram in hist.items():
            for v, c in histogram.items():
                print(f"{h_type};{v};{c}")


class Menu:
    @staticmethod
    def show_options():
        print("=" * 80)
        print("Dostępne polecenia: ")
        print("  quit        - wyjście")
        print("  lu_search   - wyszukiwanie jednostek po zadanym lu_id")
        print("  wsd_s       - ujednoznacznianie synsetami - wyszukiwanie")
        print("  wsd_lu      - ujednoznacznianie jednostkami - wyszukiwanie")
        print("  histogram   - statystyki długości lematów LU (w tokenach/znakach)")
        print("=" * 80)
        print("")


while 1:
    Menu.show_options()

    try:
        opt_menu = input("Podaj komendę: ")
        if opt_menu is None or not len(opt_menu.strip()):
            continue

        if opt_menu.lower() in ["quit", "q"]:
            break
        elif opt_menu.lower() == "lu_search":
            LuEmbeddingSearch.run_lu_search()
        elif opt_menu.lower() == "wsd_s":
            TextSearch.run_text_search(use_synset=True, use_lu=False)
        elif opt_menu.lower() == "wsd_lu":
            TextSearch.run_text_search(use_synset=False, use_lu=True)
        elif opt_menu.lower() == "histogram":
            StatsModule.historgram_lu_tokens_chars()
        else:
            Menu.show_options()
    except KeyboardInterrupt:
        break
