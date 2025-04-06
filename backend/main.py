import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from langchain_groq import ChatGroq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import ChatOpenAI
import os

app = FastAPI()

# Setup your API key and LLMs
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

llm = ChatGroq(
    temperature=0.7,
    model_name="groq/llama-3.3-70b-versatile",
    max_tokens=4000
)
chat_llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ["GROQ_API_KEY"],
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=1000,
)

# Load documents and create the index once (consider caching this for production)
documents = SimpleDirectoryReader("data").load_data()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create query engine and tool
query_engine = index.as_query_engine(similarity_top_k=5, llm=chat_llm)
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report",
)

def simulate_crew_processing(user_query: str):
    """
    This generator simulates the crew agents streaming logs.
    Replace this logic with actual callbacks or log captures from your crew agents.
    """
    # Simulated initial log
    yield f"Starting analysis for query: '{user_query}'\n"
    time.sleep(1)

    # Define agents
    researcher = Agent(
        role="Analyst",
        goal="Find insights",
        backstory="Expert in tech reports and company filings",
        verbose=True,
        tools=[query_tool],
        llm=llm,
        max_rpm=100,
    )
    writer = Agent(
        role="Content Writer",
        goal="Present insights clearly",
        backstory="Converts technical info into readable formats",
        verbose=True,
        llm=llm,
        max_rpm=100,
    )

    # Log the setup
    yield "Agents configured. Starting tasks...\n"
    time.sleep(1)

    # Define tasks
    task1 = Task(
        description=f"Analyze the query: '{user_query}'",
        expected_output="Bullet points with insights",
        agent=researcher,
    )
    task2 = Task(
        description="Convert findings into a short article for laypeople",
        expected_output="An engaging summary in 3–4 paragraphs",
        agent=writer,
    )

    yield "Tasks defined. Running crew...\n"
    time.sleep(1)

    # Create and run the crew (here, we simulate streaming by yielding messages before and after kickoff)
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
    )
    
    # Simulate processing logs
    for i in range(3):
        yield f"Processing... step {i+1}/3 completed.\n"
        time.sleep(1)
    
    # Execute the crew process (this is a blocking call in this example)
    result = crew.kickoff()
    
    # Yield the final result
    yield f"Final Result:\n{result}\n"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the query from the client
        data = await websocket.receive_text()
        user_query = data.strip()
        await websocket.send_text(f"Received query: {user_query}")
        
        # Simulate streaming response using a thread pool executor so as not to block the event loop.
        loop = asyncio.get_event_loop()
        generator = simulate_crew_processing(user_query)
        
        # Stream each message to the client as it becomes available.
        for message in generator:
            # If blocking (e.g., time.sleep) is used, run in executor:
            await loop.run_in_executor(None, lambda: None)  # Dummy call to yield control.
            await websocket.send_text(message)
        
        await websocket.send_text("Stream complete.")
    except WebSocketDisconnect:
        print("Client disconnected")
