from langchain.memory import ConversationBufferMemory
from summarize import summarize_conversation
from retrieve_context import retrieve_documents
from chat_gemini import chat_gemini
from gemini import initialize_gemini

memory = ConversationBufferMemory(return_messages=True)

def chat_with_llm(user_message: str, category: str):
    
    try:
        history = memory.load_memory_variables({})
        history_context = "\n".join([f"{m.type}: {m.content}" for m in history.get("history", [])])
        
        # Check if message is a greeting
        docs = retrieve_documents(user_message, category)
        # Format retrieved documents with metadata
        context_parts = []
        for i, doc in enumerate(docs, 1):
            doc_text = f"""Document {i}:
            Document Name: {doc.metadata['doc_name']}
            Content: {doc.page_content}
            ---"""
            context_parts.append(doc_text)
        context = "\n".join(context_parts)
        
        final_prompt = f"""You are a knowledgeable and helpful assistant for answering questions about documents.

        Your personality:
        - Professional and clear
        - Responds naturally to greetings
        - Keeps responses focused and informative

        Current conversation:
        User query: {user_message}
        Previous chat history: {history_context}

        If the user's message is a greeting (like "hi", "hello", "how are you"):
        - Respond with a polite, professional greeting
        - Keep it brief and natural
        - Don't provide any document information

        Otherwise, provide helpful information from the relevant documents with:
        1. Direct answers based on the retrieved context
        2. Clear explanations of key points
        3. Accurate information from the source material
        4. Professional tone
        5. Concise responses focused on the query
        6. Always cite the specific document names you are referencing in your response
        7. Begin each key point with "According to [Document Name]..."
        8. End your response with a list of all document sources used

        Relevant document context: 
        {context}

        Remember to explicitly mention document names when providing information.

        Response:"""

        final_response = chat_gemini(final_prompt)
        print(f"Respond message: {final_response}")

        gemini_model = initialize_gemini()        
        conversation_summary = summarize_conversation(user_message, final_response, gemini_model)
        
        memory.save_context(
            {"input": f"{user_message}"}, 
            {"output": f"{conversation_summary}"}
        )
        
        return final_response
        
    except Exception as e:
        raise Exception(f"Error in chat processing: {str(e)}")