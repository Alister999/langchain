"""Module of custom LLM module"""
import logging
import re
from typing import Any

from crewai.llm import LLM
from httpx import Client

logger = logging.getLogger("CustomLLM")


class CustomLLM(LLM):
    """Class of custom llm module for litellm"""
    def __init__(self, endpoint: str, prompt_template: str = None):
        super().__init__(model="custom", base_url=endpoint)
        self.endpoint = endpoint
        self.default_prompt_template = prompt_template or (
            "You are a helpful assistant. Answer only based on the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        self.client = Client(timeout=120.0)

    def call(self, inputs, **kwargs: Any) -> str: # pylint: disable=arguments-differ, disable=unused-argument
        logger.info("Received inputs: %s, type: %s", inputs, type(inputs))

        if isinstance(inputs, list):
            context = ""
            question = ""
            for message in inputs:
                if message.get("role") == "user" and "content" in message:
                    content = message["content"]
                    # r_string_context = r"Context:\s*(.*?)\s*Question:"
                    context_match = re.search(r"Context:\s*(.*?)\s*Question:", content, re.DOTALL)
                    r_string_match = r"Question:\s*(.*?)\s*(?:This is the expected criteria|$)"
                    question_match = re.search(r_string_match, content,
                                               re.DOTALL)
                    if context_match:
                        context = context_match.group(1).strip()
                    if question_match:
                        question = question_match.group(1).strip()
            inputs = {"context": context, "question": question}
        elif not isinstance(inputs, dict):
            raise ValueError(f"Inputs must be a dictionary or a list, got {type(inputs)}")

        logger.info("Processed inputs: %s", inputs)

        prompt_template = inputs.get("prompt_template", self.default_prompt_template)

        prompt = prompt_template.format(
            context=inputs.get("context", ""),
            question=inputs.get("question", "")
        ) if "{context}" in prompt_template or "{question}" in prompt_template else prompt_template
        logger.info("Formatted prompt: %s", prompt)

        try:
            response = self.client.post(
                self.endpoint,
                json={"prompt": prompt}
            )
            response.raise_for_status()
            # result = response.json()
            logger.info("Response: %s", response.json())

            answer = response.json().get("content", response.json().get("completion", ""))
            if not answer:
                logger.error("No valid answer found in response: %s", response.json())
                raise ValueError("No valid answer found in LLM response")

            answer = answer.strip()
            logger.info("Extracted answer: %s", answer)
            return answer

        except TimeoutError as e:
            logger.error("Error calling endpoint: %s", e, exc_info=True)
            raise

        except Exception as e:
            logger.error("Error calling endpoint: %s", e, exc_info=True)
            raise
