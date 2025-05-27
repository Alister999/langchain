from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


texts = [
    "Пиво: Хеникен 4 бутылки 0.5л холодильник - Beer: Heineken 4 bottle 0.5l refrigerator",
    "Пиво: Туборг 6 бутылок 0.5л холодильник - Beer: Tuborg 6 bottle 0.5l refrigerator",
    "Пиво: Будвайзер 3 бутылки 0.5л полка - Beer: Budweiser 3 bottle 0.5l shelve",
    "Безалкогольный напиток: Кока-Кола 7 бутылок 0.33л холодильник - Non alcohol: Coca-Cola 7 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Спрайт 8 бутылок 0.33л холодильник - Non alcohol: Sprite 8 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Фанта 5 бутылок 0.33л полка - Non alcohol: Fanta 5 bottle 0.33l shelve",
]

def save_embedding():
    db = FAISS.from_texts(texts, embedding_model)

    db.save_local("faiss_index")


# @tool
def get_embedding(input: str) -> str :
    """Getting contest from FAISS according incoming request."""
    try:
        print("DEBUG INPUTS:", input)
        db = FAISS.load_local("faiss_index", embedding_model,
                              allow_dangerous_deserialization=True)
        query = input
        results = db.similarity_search(query)
        final_result = [r.page_content for r in results]
        return "\n".join(final_result)
    except Exception as e:
        print("❌ Error in get_embedding:", e)
        return "ERROR"

