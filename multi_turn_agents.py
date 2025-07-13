from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import asyncio

# 1. Define Pydantic Models for Communication

# Message format for each turn in the conversation
class ConversationMessage(BaseModel):
    sender: str = Field(..., description="The name of the agent sending the message")
    content: str = Field(..., description="The content of the message for this turn")

# Output from the User Agent, including the query and a place to store the Support Agent's response
class UserAgentOutput(BaseModel):
    user_query: str = Field(..., description="The user's query or statement")
    support_response: str = Field(None, description="The response received from the Support Agent")

# Output from the Support Agent
class SupportAgentOutput(BaseModel):
    response: str = Field(..., description="The support agent's response")
    escalate_to_specialist: bool = Field(False, description="True if the issue needs escalation")

# 2. Define the Pydantic AI Agents

# User Agent
user_agent = Agent(
    model='openai:gpt-4o',  # Or your preferred LLM
    output_type=UserAgentOutput,
    system_prompt=(
        "You are a user experiencing a technical issue. You need to explain "
        "your problem clearly and respond to the support agent's questions."
    ),
)

# Support Agent
support_agent = Agent(
    model='openai:gpt-4o',  # Or your preferred LLM
    deps_type=ConversationMessage,  # Expects a message from the user
    output_type=SupportAgentOutput,
    system_prompt=(
        "You are a helpful and patient support agent. Your goal is to understand "
        "the user's problem and provide solutions. If you can't solve it, "
        "you should escalate."
    ),
)

# 3. Simulate the Multi-Turn Conversation

async def multi_turn_conversation():
    conversation_history = []  # To store the message history for context
    
    # --- Turn 1: User initiates the problem ---
    user_input_1 = "My internet is not working. I can't access any websites."
    print(f"User: {user_input_1}")
    
    # User Agent runs, generating the initial query
    user_turn_1_result = await user_agent.run(
        user_input_1,
        message_history=conversation_history  # Pass current history
    )
    # Add the user's message to the history
    conversation_history.extend(user_turn_1_result.new_messages()) #

    # Support Agent processes the user's query
    support_input_1 = ConversationMessage(sender="User", content=user_turn_1_result.output.user_query)
    support_turn_1_result = await support_agent.run(
        support_input_1,
        message_history=conversation_history # Pass current history to support agent
    )
    print(f"Support: {support_turn_1_result.output.response}")
    # Add the support agent's response to the history
    conversation_history.extend(support_turn_1_result.new_messages()) #

    # --- Turn 2: User provides more info based on support's question ---
    user_input_2 = "Yes, I've tried restarting my router multiple times, but it didn't help."
    print(f"User: {user_input_2}")
    
    # User Agent runs, responding to support
    user_turn_2_result = await user_agent.run(
        user_input_2,
        message_history=conversation_history  # Pass updated history
    )
    conversation_history.extend(user_turn_2_result.new_messages()) #

    # Support Agent processes the user's follow-up
    support_input_2 = ConversationMessage(sender="User", content=user_turn_2_result.output.user_query)
    support_turn_2_result = await support_agent.run(
        support_input_2,
        message_history=conversation_history # Pass updated history
    )
    print(f"Support: {support_turn_2_result.output.response}")
    conversation_history.extend(support_turn_2_result.new_messages()) #

    # --- Turn 3: User confirms/Support escalates if needed ---
    if support_turn_2_result.output.escalate_to_specialist:
        print("Support: This issue requires a specialist. I'm escalating your case.")
        # Optionally, a new agent for escalation could be invoked here
    else:
        user_input_3 = "The Wi-Fi light is green, but the internet light is off."
        print(f"User: {user_input_3}")

        user_turn_3_result = await user_agent.run(
            user_input_3,
            message_history=conversation_history
        )
        conversation_history.extend(user_turn_3_result.new_messages())

        support_input_3 = ConversationMessage(sender="User", content=user_turn_3_result.output.user_query)
        support_turn_3_result = await support_agent.run(
            support_input_3,
            message_history=conversation_history
        )
        print(f"Support: {support_turn_3_result.output.response}")
        conversation_history.extend(support_turn_3_result.new_messages())

# Run the conversation
if __name__ == "__main__":
    asyncio.run(multi_turn_conversation())
