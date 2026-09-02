import logfire
from langchain_groq import ChatGroq
from app.agents.state import AgentState
from app.config import settings


# Initialize the Groq model
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0.1
)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """
    query = state["current_query"]
    
    # Format the entire history for the LLM
    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    
    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.
        
        CONVERSATION HISTORY:
        {history_str}
        
        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        # Technical RAG Logic with Token Safety
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000 
        full_context = ""
        
        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        # Safety check: if retrieval returned nothing, do NOT let the LLM
        # improvise an answer. Return an honest "no info found" response
        # instead of risking a hallucinated one.
        if not full_context.strip():
            logfire.warning("No documents retrieved — skipping LLM call to avoid hallucination.")
            honest_answer = (
                "I couldn't find any relevant information in the knowledge base "
                "to answer that question. This could mean the documents haven't "
                "been indexed yet, or the retrieval system is temporarily "
                "unavailable. Please try rephrasing your question, or check "
                "that the relevant documents have been ingested."
            )
            return {
                "final_answer": honest_answer,
                "status": "No documents retrieved.",
                "messages": [{"role": "assistant", "content": honest_answer}]
            }

        prompt = f"""
        You are a Senior Technical Architect. 
        Answer the question using ONLY the TECHNICAL CONTEXT provided below.
        Do not use outside knowledge or make assumptions about content that
        is not present in the context. If the context does not contain
        enough information to answer, say so explicitly instead of guessing.
        
        TECHNICAL CONTEXT:
        {full_context}
        
        CONVERSATION HISTORY:
        {history_str}
        
        USER QUESTION:
        "{user_msg}"
        """
    
    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = llm.invoke(prompt)
            logfire.info("Response synthesized successfully.")
            return {
                "final_answer": response.content,
                "status": "Response generated.",
                "messages": [{"role": "assistant", "content": response.content}]
            }
        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e