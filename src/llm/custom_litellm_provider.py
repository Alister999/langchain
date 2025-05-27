from crewai.llm import LLM
from httpx import Client
import logging
import re

logger = logging.getLogger("CustomLLM")


class CustomLLM(LLM):
    def __init__(self, endpoint: str, prompt_template: str = None):
        super().__init__(model="custom", base_url=endpoint)
        self.endpoint = endpoint
        self.default_prompt_template = prompt_template or (
            "You are a helpful assistant. Answer only based on the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        self.client = Client(timeout=600.0)

    def call(self, inputs, **kwargs) -> str:
        logger.info(f"Received inputs: {inputs}, type: {type(inputs)}")

        if isinstance(inputs, list):
            context = ""
            question = ""
            for message in inputs:
                if message.get("role") == "user" and "content" in message:
                    content = message["content"]
                    context_match = re.search(r"Context:\s*(.*?)\s*Question:", content, re.DOTALL)
                    question_match = re.search(r"Question:\s*(.*?)\s*(?:This is the expected criteria|$)", content,
                                               re.DOTALL)
                    if context_match:
                        context = context_match.group(1).strip()
                    if question_match:
                        question = question_match.group(1).strip()
            inputs = {"context": context, "question": question}
        elif not isinstance(inputs, dict):
            raise ValueError(f"Inputs must be a dictionary or a list of messages, got {type(inputs)}")

        logger.info(f"Processed inputs: {inputs}")

        prompt_template = inputs.get("prompt_template", self.default_prompt_template)

        # Формируем промпт
        prompt = prompt_template.format(
            context=inputs.get("context", ""),
            question=inputs.get("question", "")
        ) if "{context}" in prompt_template or "{question}" in prompt_template else prompt_template
        logger.info(f"Formatted prompt: {prompt}")

        try:
            response = self.client.post(
                self.endpoint,
                json={"prompt": prompt}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response: {result}")

            answer = result.get("content", result.get("completion", ""))
            if not answer:
                logger.error(f"No valid answer found in response: {result}")
                raise ValueError("No valid answer found in LLM response")

            answer = answer.strip()
            logger.info(f"Extracted answer: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Error calling endpoint: {e}", exc_info=True)
            raise