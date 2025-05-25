import threading
import logging
from datetime import datetime
import requests.exceptions
from src.llm.local_model import LocalModel
from src.utils.utils import show_spinner

logging = logging.getLogger("ChatLogger")


def go_chat(model: LocalModel):
    logging.info("Incoming to go chat func")
    while True:
        my_input = input("Enter you prompt: ")
        start_time = datetime.now()

        if my_input == "stop" or my_input == 'enough':
            break

        prompt = f"<|im_start|>user\\n{my_input}<|im_end|>\\n<|im_start|>assistant\\n"

        stop_event = threading.Event()
        progress_thread = threading.Thread(target=show_spinner, args=(stop_event,))
        progress_thread.start()

        try:
            # Запускаем модель и ждём ответа
            response = model.invoke(prompt)  # llm(prompt)
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error of connection - {e}")
            stop_event.set()
            break

        # Останавливаем прогресс бар
        stop_event.set()
        progress_thread.join()

        stop_time = datetime.now()
        logging.info(f"Time of working request - {stop_time - start_time}s")

        # return response
        print(response)