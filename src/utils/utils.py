import sys
import time
import logging

logging = logging.getLogger("UtilsLogger")


def show_spinner(stop_event):
    logging.info("Coming to spinner func")
    spinner = ['*', '**', '***', '****']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rОжидание ответа... {spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.7)

    logging.info("Operation complete.")
    sys.stdout.write("\rОбработка завершена.          \n")

