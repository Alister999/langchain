import threading
import logging
from datetime import datetime
import requests.exceptions

from src.core.faiss_db import get_embedding
from src.llm.local_model import LocalModel
from src.utils.utils import show_spinner, create_prompt

logging = logging.getLogger("ChatLogger")


def go_chat(model: LocalModel):
    logging.info("Incoming to go chat func")
    while True:
        my_input = input("Enter you prompt: ")
        start_time = datetime.now()

        if my_input == "stop" or my_input == 'enough':
            break

        context = get_embedding(prompt=my_input)
        logging.info(f"getting context {context}")

        stop_event = threading.Event()
        progress_thread = threading.Thread(target=show_spinner, args=(stop_event,))
        progress_thread.start()


        try:
            # Запускаем модель и ждём ответа
            final_prompt = create_prompt(context=context, answer=my_input)

            logging.info(f'Prompt request - {final_prompt}')
            response = model.invoke(final_prompt)
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error of connection - {e}")
            stop_event.set()
            break

        # Останавливаем прогресс бар
        stop_event.set()
        progress_thread.join()

        stop_time = datetime.now()
        logging.info(f"Time of working request - {stop_time - start_time}s")

        print(response)