import streamlit as st
from streamlit_chat import message
from llm_model import *
#from llm_model import google_model, generic_google_chain, google_chain, movies_vectorstore, recipies_vectorstore, cur_emotion, query_number, update_emotion, update_query_number, memory, emotion_classifier_chain
import textwrap

with open("static_init.txt", "r") as fp:
    init = int(fp.read())
def printer():
    with open("static_init.txt", "w") as fp:
        fp.write("1")
    print("hello")

def load_llm():
    pass

if(init == 0):
    printer()
    load_llm()


def invoke_chain(google_docs, query):
    query = f"user emotion is : {cur_emotion[2:]} \n query : " + query
    print("rag model ================", query)
    try:
       response = google_chain({"input_documents": google_docs, "question": query}, return_only_outputs=True)
       return response['output_text']
    except:
       return "sorry I cannot answer that"

def invoke_model(query_prompt):
    query_prompt = f"user emotion is : {cur_emotion[2:]} \n query : " + query_prompt
    print("generic model ===================", query_prompt)
    try:
       response = generic_google_chain(query_prompt)
       #print("generic response : ", response)
       return response['text']
    except:
       return "sorry I cannot answer that"

def classify_query(query):
    query_prompt = f"tell me in one word if this question is related to movies or cooking or other\n\n query : {query}"
    llm_text = google_model.invoke(query_prompt).content
    print("============query classified as ==================", llm_text)
    return llm_text.lower()

def llm_text_classify(statement):
    print(statement)
    compare_text = "answer not provided in the context"
    query_prompt = f"tell me yes or no if statement1 and statement2 have similar meanings \n statement1 : {statement} \n statement2 : {compare_text}"
    print(query_prompt)
    llm_text = google_model.invoke(query_prompt).content
    print(llm_text)
    return llm_text.lower()

def classify_emotion():
    query = f"""based on the chat history, tell which of the following emotion state closely matches the user emotion \n
            1)happy \n 2)sad \n 3)nervous \n 4)confused \n 5)angry"""
    print("emotion classifier invoked")
    try :
        response = emotion_classifier_chain(query)
        print("emotion classified as : ", response['text'])
        return response['text']
    except:
        return "nothing"
      

def api_calling(query):
    if((query_number % 1) == 0):
       emotion = classify_emotion()
       memory.chat_memory.messages = memory.chat_memory.messages[:-2]
       update_emotion(emotion)
    query_type = classify_query(query)
    print("================= query number : ", query_number)
    update_query_number()
    if(query_type == "movies"):
       print("movies model")
       google_docs = movies_vectorstore.similarity_search(query, k = 6)
       llm_text = invoke_chain(google_docs, query)
    elif(query_type == "cooking"):
       print("cooking model")
       google_docs = recipies_vectorstore.similarity_search(query, k = 6)
       llm_text = invoke_chain(google_docs, query)
    else:
       #query_prompt = f"reply with empathy for the question \n\n query : {query}"
       query_prompt = query
       llm_text = invoke_model(query_prompt)

    if(llm_text == "sorry I cannot answer that"):       #this reply comes only when an exception is caught so we will return as is
       llm_text = llm_text
    else:
       is_llm_response_invalid = llm_text_classify(llm_text)
    
       if("yes" in is_llm_response_invalid):
          memory.chat_memory.messages = memory.chat_memory.messages[:-1]
          #query_prompt = f"answer the question in a friendly conversational manner \n\n query : {query}"
          query_prompt = query
          llm_text = invoke_model(query_prompt)

    final_answer = textwrap.dedent(llm_text).strip()
    return final_answer

st.title("Emphathetic chatbot")

if 'user_input' not in st.session_state:
	st.session_state['user_input'] = []

if 'model_response' not in st.session_state:
    st.session_state['model_response'] = []

def get_text():
	input_text = st.text_input("write text", key="input")
	return input_text

user_input = get_text()
#print("hello")
if user_input:
	output = api_calling(user_input)
	output = output.lstrip("\n")

	# Store the output
	st.session_state.model_response.append(user_input)
	st.session_state.user_input.append(output)

message_history = st.empty()

if st.session_state['user_input']:
	for i in range(len(st.session_state['user_input']) - 1, -1, -1):
		# This function displays user input
		message(st.session_state["user_input"][i],
				key=str(i),avatar_style="icons")
		# This function displays OpenAI response
		message(st.session_state['model_response'][i],
				avatar_style="miniavs",is_user=True,
				key=str(i) + 'data_by_user')


