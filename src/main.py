import logging
from src.chats.chat_regular import go_chat
from src.core.logger_core import setup_logging
from src.llm.local_model import LocalModel

setup_logging()
logging = logging.getLogger("MainLogger")

model = LocalModel()

logging.info("Coming to program and go chat func")
go_chat(model)
logging.info("Finish of program")