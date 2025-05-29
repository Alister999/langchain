import logging
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

logger = logging.getLogger("MivinusLogger")


write_texts = [
    "Пиво: Хеникен 4 бутылки 0.5л холодильник - Beer: Heineken 4 bottle 0.5l refrigerator",
    "Пиво: Туборг 6 бутылок 0.5л холодильник - Beer: Tuborg 6 bottle 0.5l refrigerator",
    "Пиво: Будвайзер 3 бутылки 0.5л полка - Beer: Budweiser 3 bottle 0.5l shelve",
    "Пиво: Холстен 3 бутылки 0.5л полка - Beer: Holsten 3 bottle 0.5l shelve",
    "Безалкогольный напиток: Кока-Кола 7 бутылок 0.33л холодильник - Non alcohol: Coca-Cola 7 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Спрайт 8 бутылок 0.33л холодильник - Non alcohol: Sprite 8 bottle 0.33l refrigerator",
    "Безалкогольный напиток: Фанта 5 бутылок 0.33л полка - Non alcohol: Fanta 5 bottle 0.33l shelve",
]

# Подключение
connections.connect(host="localhost", port="19530")

# Определение схемы
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Размерность эмбеддингов
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields=fields, description="Text embeddings")
collection = Collection(name="items", schema=schema)
# Модель эмбеддингов
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def write_data(texts: list):
    # texts = ["Пиво: Будвайзер 3 бутылки 0.5л", "Пиво: Хеникен 4 бутылки 0.5л"]
    embeddings = embedding_model.embed_documents(texts)
    ids = list(range(len(texts)))

    # Вставка данных
    collection.insert([ids, embeddings, texts])


    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()  # Загрузка коллекции в память


def get_data(query_text: str):
# query_text = "Пиво Будвайзер"
    query_embedding = embedding_model.embed_query(query_text)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 100}},
        limit=5,
        output_fields=["content"]
    )
    for hits in results:
        for hit in hits:
            print(f"ID: {hit.id}, Content: {hit.entity.get('content')}, Distance: {hit.distance}")


def update_data(new_texts: str, texts: list):
    # new_texts = ["Пиво: Туборг 6 бутылок 0.5л"]
    new_embeddings = embedding_model.embed_documents(new_texts)
    new_ids = [len(texts)]  # Новые ID
    collection.insert([new_ids, new_embeddings, new_texts])



collection.delete("id IN [0]")