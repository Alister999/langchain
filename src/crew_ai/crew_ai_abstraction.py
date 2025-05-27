from crewai import Agent, Crew, Task

from src.core.prompt_config import bar_prompt, joke_prompt
from src.llm.custom_litellm_provider import CustomLLM


my_llm = CustomLLM(endpoint="http://192.168.100.14:8000/completion")

llm_agent = Agent(
    llm=my_llm,
    role=" ",
    goal=" ",
    backstory="Nothing",
    verbose=True,
)

task = Task(
    description= (
        "Context: {context}\n\n"
        "Question: {request}"
    ),
    agent=llm_agent,
    prompt_template= joke_prompt, #bar_prompt,
    expected_output="A concise and accurate answer based strictly on the provided context.",
)

crew = Crew(
    agents=[llm_agent],
    tasks=[task],
    verbose=True,
)



