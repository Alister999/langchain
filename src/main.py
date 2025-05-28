import logging
from datetime import datetime

from src.core.faiss_db import save_embedding, get_embedding
from src.core.logger_core import setup_logging
from src.crew_ai.crew_ai_abstraction import crew

setup_logging()
logging = logging.getLogger("MainLogger")

logging.info("Coming to program and go chat func")

def main():
    print("Hello! Say your question (введи 'выход' для выхода):")
    start_time = datetime.now()
    while True:
        user_input = input("\n❓ Вопрос: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("👋 Пока!")
            break
        try:
            context = get_embedding(input=user_input)

            logging.warning(f"Users input - {user_input}")
            logging.warning(f"Users context - {context}")

            inputs = {
                "request": user_input,
                "context": context
            }
            logging.info(f"Inputs for crew.kickoff: {inputs}, type: {type(inputs)}")
            result = crew.kickoff(inputs=inputs)
            print(f"\n🧠 Ответ: {result}")
            finish_time = datetime.now()
            logging.warning(f"Time of request - {finish_time - start_time}min")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    save_embedding()
    main()

logging.info("Finish of program")