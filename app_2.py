from dotenv import load_dotenv
import os 
import streamlit as st
from langchain_community.document_loaders import  PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq 
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
load_dotenv()

# Read OpenAI API key securely (Streamlit Cloud Secrets)
openai_api_key = os.getenv("OPENAI_API_KEY")
grOq_api_key = os.getenv("GROQ_API_KEY")



if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_documents():
    # Use PyPDFLoader instead of TextLoader
    loader = PyPDFLoader("indusmedical.pdf") 
    pages = loader.load_and_split()  

    emb_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(pages, emb_model)
    return vectorstore

# Initialize the vectorstore
vectorstore = load_documents()


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.3)

# Updated Prompt Template
prompt = PromptTemplate(
    template="""**Role**: You are an AI assistant for a Retrieval-Augmented Generation (RAG) system**.

### **Instructions**:
1. **Source of Truth**:
   - **Strictly use only the provided `{data}`** to answer questions about indus medical.
   - If the answer isn't in the data, respond:  
     ❌ "I'm sorry, I don't have that information."
   - For follow-up questions, use the chat history to maintain context.

2. **Allowed Free Responses**:
   - Greetings, farewells, and general small talk are fine without relying on data.

3. **Response Style**:
   - Be professional, helpful, and concise.
   - Ensure responses are user-friendly and clear.

### **Chat History (Last 4 messages) reply according to the chat history and from given data if user are asking about anything related to previous messages you know you have to continue the conversation**:
{chat_history}

### **User's Latest Question**:
{query}

### **Retrieved Data**:
{data}

### **Assistant's Response**:""",
    input_variables=["data", "query", "chat_history"]
)

# Format retrieved documents
def doc_format(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

# Format chat history using AIMessage and HumanMessage
def get_chat_history(_):
    messages = st.session_state.get("messages", [])
    # Take the last 4 messages or fewer if not enough
    last_msgs = messages[-4:] if len(messages) > 0 else []
    history = []
    for msg in last_msgs:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    # Convert messages to a string representation for the prompt
    history_str = ""
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    return history_str.strip()

# LangChain Pipeline
parallel_chain = RunnableParallel({
    "data": retriever | RunnableLambda(doc_format),
    "query": RunnablePassthrough(),
    "chat_history": RunnableLambda(get_chat_history)
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser


st.markdown(
    """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;">
        <h1 style="text-align: center;  text-size: 30px">INDUS MEDCIAL AI Assistant</h1>
        <p style="text-align: center; font-size: 1.2em;">
            Welcome to the official chatbot of INDUS MEDICAL.<br>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# Display previous messages
for message  in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input and response
if user_input := st.chat_input("Ask about INDUS MEDICAL, services, or anything else..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = main_chain.invoke(user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                response = "I'm sorry, something went wrong. Please try again."
        
        st.session_state.messages.append({"role": "assistant", "content": response})



