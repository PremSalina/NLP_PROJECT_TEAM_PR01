from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import textwrap
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.chains import LLMChain


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
movies_vectorstore = FAISS.load_local("/home/sandeep/Downloads/movies_faiss_index", embeddings, allow_dangerous_deserialization = True)
recipies_vectorstore = FAISS.load_local("/home/sandeep/Downloads/recipies_faiss_index", embeddings, allow_dangerous_deserialization = True)

print("vectorstores loaded =====================")
query = "what is the summary of toy story"
#docs_and_scores = vectorstore.similarity_search(query, 5)
#print(docs_and_scores)


googleApiKey = "AIzaSyBJIfyFYdLMRA95M3DYwtFiuSl55VrDsw4"

google_model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3, google_api_key = googleApiKey)

prompt_template = """ you are a chat bot having conversation with human. Using the following documents and chat history, answer the question in a friendly conversational manner empathetically to sync the emotion of user given in Question.\n\n
    Context: {context}?\n
    Chat_history : {chat_history}\n
    Question: {question}\n

    Answer:
    """

google_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=googleApiKey)
google_prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])

memory  = ConversationBufferMemory(memory_key = "chat_history", input_key = "question")
google_chain = load_qa_chain(google_model, chain_type="stuff", prompt=google_prompt, memory = memory)


generic_template = """You are a chatbot having a conversation with a human. Reply with general correct answers, empahtetically to sync with the emotion given in Question.\n

{chat_history}
Question : {question}
Chatbot:"""

generic_google_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=generic_template
)


generic_google_chain = LLMChain(
    llm=google_model,
    prompt=generic_google_prompt,
    verbose=False,
    memory=memory,
)

emotion_classifier_template = """ Based on the chat history, you have to classify user emotion\n

{chat_history}
Question : {question}
Chatbot:"""

emotion_classifier_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=emotion_classifier_template
)


emotion_classifier_chain = LLMChain(
    llm=google_model,
    prompt=emotion_classifier_prompt,
    verbose=False,
    memory=memory,
)



query_number = 1
#with open("static_init.txt", "r") as fp:
#    query_number = int(fp.read())
def update_query_number():
    global query_number
    query_number += 1
print(query_number)

cur_emotion = "nothing"

def get_emotion():
    global cur_emotion
    return cur_emotion

def update_emotion(emotion):
    global cur_emotion
    cur_emotion = emotion

