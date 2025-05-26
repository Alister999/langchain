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

def create_prompt(context: list, answer: str) -> str:
    # You are a helpful assistant. Always answer strictly based on the provided context. Be brief and do not invent facts. If no answer is found in the context, say: "No information available."
    prompt = f"""<|system|>
    You are a helpful assistant. You always answer based on the context. Try to extract all possible useful information. Be concise.
    <|user|>
    Context:
    {chr(10).join(context)}
    Question: {answer}
    <|end|>
    <|assistant|>"""

    return prompt