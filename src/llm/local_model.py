from typing import Optional, List
import logging
import requests
from langchain_core.language_models import LLM

logging = logging.getLogger("ModelLogger")


class LocalModel(LLM):
    endpoint: str = "http://192.168.100.14:8000/completion"
    stop_tokens: Optional[List[str]] = ["<|im_end|>"]
    temperature: float = 0.7
    n_predict: int = 32 #128

    def _call(self):
        pass

    def invoke(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logging.info("Calling invoke func")
        payload = {
            "prompt": prompt,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "stop": stop or self.stop_tokens,
        }

        logging.info("Sending post request to server")
        response = requests.post(self.endpoint, json=payload)
        if response.status_code != 200:
            logging.warning(f"Request failed: {response.status_code} - {response.text}")
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        logging.info("Request was successful")
        return response.json()["content"]

    @property
    def _identifying_params(self):
        return {"endpoint": self.endpoint}

    @property
    def _llm_type(self) -> str:
        return "local"