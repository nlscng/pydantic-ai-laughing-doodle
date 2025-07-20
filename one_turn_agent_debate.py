from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult

supportive_agent = Agent(
    "openai:gpt-4o",
    system_prompt="Your name is John, you are in a debate, "
    "and you are debating from the stand point of supporting the debate topic.",
)
critical_agent = Agent(
    "openai:gpt-4o",
    system_prompt="Your name is Jack, you are in a debate, "
    "and you are debating from the stand point of be critical of the debate topic.",
)

debate_topic = "Is AI a threat to humanity?"
result1: AgentRunResult[str] = supportive_agent.run_sync(
    f"Start the debate! Our topic today is {debate_topic}, John, please begin."
)
print(result1.output)
# > Did you hear about the toothpaste scandal? They called it Colgate.

result2 = critical_agent.run_sync(
    f"Jack, you are up, please begin.", message_history=result1.new_messages()
)
print(result2.output)
# > This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
