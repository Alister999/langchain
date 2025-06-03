"""General module, entrypoint for app"""
import logging
from datetime import datetime
from http.client import HTTPException

from src.core.faiss_db import save_embedding, get_embedding
from src.core.logger_core import setup_logging
from src.core.mivinus_db import write_data, get_data
from src.crew_ai.crew_ai_abstraction import crew

setup_logging()
logger = logging.getLogger("MainLogger")

logger.info("Coming to program and go chat func")

def main():
    """General function for running main thread"""
    print("Hello! Say your question (введи 'выход' для выхода):")
    start_time = datetime.now()
    while True:
        user_input = input("\n❓ Question: ")
        if user_input.lower() in ["выход", "exit", "quit", "stop"]:
            print("👋 Bye!")
            break
        try:
            context = get_data(query_text=user_input) #get_embedding(input=user_input)

            logger.warning("Users input - %s", user_input)
            logger.warning("Users context - %s", context)

            inputs = {
                "request": user_input,
                "context": context
            }
            logger.info("Inputs for crew.kickoff: %s, type: %s", inputs, type(inputs))
            result = crew.kickoff(inputs=inputs)

            print(f"\n🧠 Answer: {result}")

            finish_time = datetime.now()
            logger.warning("Time of request - %s min", finish_time - start_time)
        except KeyboardInterrupt as e:
            logger.warning("Keyboard input was interrupted - %s", e)
        except ValueError as e:
            logger.error("There is value exception - %s", e)
        except HTTPException as e:
            logger.error("HTTP exception - %s", e)
        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)
            raise

if __name__ == "__main__":
    # save_embedding()
    write_data()
    main()

logger.info("Finish of program")
