# Import necessary modules and setup for FastAPI, LangGraph, and LangChain
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import uvicorn

# LangChain / LangGraph imports
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------------------------------------------------
# API KEYS
# --------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = os.getenv(
    "GOOGLE_API_KEY",
    "AIzaSyDgYzv6MEud392oCkAX96BlCeqa69wa-xM"
)

os.environ["TAVILY_API_KEY"] = os.getenv(
    "TAVILY_API_KEY",
    "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"
)

# --------------------------------------------------------------------
# SUPPORTED GEMINI MODELS
# --------------------------------------------------------------------
MODEL_NAMES = [
    "gemini-2.5-flash"
]

# --------------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------------
tool_tavily = TavilySearchResults(max_results=2)
tools = [tool_tavily]

# --------------------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------------------
app = FastAPI(title="LangGraph Gemini AI Agent")

# --------------------------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------------------------
class RequestState(BaseModel):
    system_prompt: str
    model_name: str
    messages: List[str]

# --------------------------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    Chat endpoint using Gemini + LangGraph ReAct agent
    """

    if request.model_name not in MODEL_NAMES:
        return {
            "error": f"Invalid model name. Choose from {MODEL_NAMES}"
        }

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=request.model_name,
        temperature=0.2
    )

    # Create ReAct Agent
    agent = create_react_agent(
    model=llm,
    tools=tools,
    # state_modifier=request.system_prompt
)


    # Initial agent state
    state = {
        "messages": request.messages
    }

    # Invoke agent
    result = agent.invoke(state)

    return result

# --------------------------------------------------------------------
# RUN SERVER
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
