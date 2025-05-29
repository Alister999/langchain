import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger("FaissLogger")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [
    "Пиво: Хеникен 4 бутылки 0.5л холодильник - Beer: Heineken 4 bottle 0.5l refrigerator",
    "Пиво: Туборг 6 бутылок 0.5л холодильник - Beer: Tuborg 6 bottle 0.5l refrigerator",
    "Пиво: Будвайзер 3 бутылки 0.5л полка - Beer: Budweiser 3 bottle 0.5l shelve",
    "Пиво: Холстен 3 бутылки 0.5л полка - Beer: Holsten 3 bottle 0.5l shelve",
    "Безалкогольный напиток: Кока-Кола 7 бутылок 0.33л холодильник - Non alcohol: Coca-Cola 7 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Спрайт 8 бутылок 0.33л холодильник - Non alcohol: Sprite 8 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Фанта 5 бутылок 0.33л полка - Non alcohol: Fanta 5 bottle 0.33l shelve",
]


def save_embedding():
    """Сохраняет FAISS index в указанную директорию."""
    dir = "./data/embeddings"
    try:
        os.makedirs(dir, exist_ok=True)
        index_path = os.path.join(dir, "index.faiss")

        if os.path.exists(index_path): # and not force_recreate:
            logger.info(f"Loading existing FAISS index from {dir}")
            db = FAISS.load_local(dir, embedding_model,
                              allow_dangerous_deserialization=True)
            existing_texts = {doc.page_content for doc in db.docstore._dict.values()}
            new_texts = [t for t in texts if t not in existing_texts]
            if new_texts:
                logger.info(f"Adding {len(new_texts)} new texts to index")
                db.add_texts(new_texts)
                db.save_local(dir)
                logger.info(f"Updated FAISS index saved to {dir}")
            else:
                logger.info("No new texts to add")
        else:
            logger.info(f"Creating new FAISS index in {dir}")
            db = FAISS.from_texts(texts, embedding_model)
            db.save_local(dir)
            logger.info(f"FAISS index saved to {dir}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
        raise



def get_embedding(input: str) -> str :
    """Getting contest from FAISS according incoming request."""
    dir = "./data/embeddings"
    try:
        print("DEBUG INPUTS:", input)
        db = FAISS.load_local(dir, embedding_model,
                              allow_dangerous_deserialization=True)
        query = input
        results = db.similarity_search(query)
        final_result = [r.page_content for r in results]
        return "\n".join(final_result)
    except Exception as e:
        print("❌ Error in get_embedding:", e)
        return "ERROR"

