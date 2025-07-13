from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import asyncio

# Define the structure of the message passed between agents
class AgentMessage(BaseModel):
    sender: str = Field(..., description="The name of the agent sending the message")
    content: str = Field(..., description="The content of the message")

class ResponderResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")

# Define the Responder agent
responder_agent = Agent(
    model='openai:gpt-4o',  # Or your chosen LLM
    deps_type=AgentMessage,  # Expects a message from the questioner
    output_type=ResponderResponse,
    system_prompt="You are a helpful assistant. Provide concise and accurate answers to questions."
)

class QuestionerResponse(BaseModel):
    query: str = Field(..., description="The question asked by the agent")
    response_from_responder: str = Field(..., description="The response received from the Responder agent")

# Define the Questioner agent
questioner_agent = Agent(
    model='openai:gpt-4o',  # Or your chosen LLM
    output_type=QuestionerResponse,
    system_prompt="You are a curious questioner. Ask a question and wait for an answer."
)

# Simulate the conversation
async def main():
    # Questioner agent initiates the conversation
    question_query = "What is the capital of France?"
    
    # Questioner agent asks the question
    question_result = await questioner_agent.run(question_query)
    
    # Responder agent receives the question and generates a response
    responder_input = AgentMessage(sender="Questioner", content=question_result.output.query)
    responder_result = await responder_agent.run(responder_input)
    
    # Questioner agent incorporates the response from the Responder
    final_questioner_response = QuestionerResponse(
        query=question_result.output.query,
        response_from_responder=responder_result.output.answer
    )
    
    print(f"Questioner asked: {final_questioner_response.query}")
    print(f"Responder answered: {final_questioner_response.response_from_responder}")

if __name__ == "__main__":
    asyncio.run(main())
