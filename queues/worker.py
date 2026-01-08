from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types

#setting up gemini client
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="company",
    embedding=embedding_model
)
def passing():
    return "passed to owner"
class ToolDetails(BaseModel):
    isToolCall: Literal["yes", "no"] = Field(description="if the ai agent has to pass the user query to the owner as the data is not in context, so if it has to call the tool or not")
    ans: str = Field(description="The Answer/ response to the user query")

class Answer(BaseModel):
    ans: str = Field(description="The Answer/ response to the user query")
class ModerationResult(BaseModel):
    decision: ToolDetails

def process_query(query:str):
    print("Searching Chunks")
    search_results = vector_db.similarity_search(
        k=8,
        query=query
    )
    contexts = []

    for result in search_results:
        contexts.append(
        f"""
        Page Content:
        {result.page_content}

        Page Number: {result.metadata.get('page_label')}
        Source File: {result.metadata.get('source')}
           """
    )
        context = "\n\n---\n\n".join(contexts)

    SYSTEMPROMPT = f"""
You are the official AI CRM agent for Consolation Furnishings.
Your job is to:
- Answer customer questions accurately using company knowledge
- Be polite, professional, and concise
- Never hallucinate prices or policies
- Ask clarifying questions when needed
- Capture leads by asking name, product interest, and location
- Escalate to human support if unsure

Context:
{context}
If information is not found in knowledge base, say:
"Let me connect you with our team for accurate details."

"""
    
    response = gemini_client.models.generate_content(
    model="gemini-2.0-flash",
    contents=query,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": ModerationResult.model_json_schema(),
        "system_instruction":SYSTEMPROMPT
    },
)

    recipe = ModerationResult.model_validate_json(response.text)
    if recipe.decision.isToolCall == "yes":
        print(passing())
        return recipe.decision.ans
    elif recipe.decision.isToolCall == "no":
        return recipe.decision.ans