# NLP_PROJECT_TEAM_PR01
llm for social good, an empathetic chatbot

steps to run milestone 3:

1. Load Vectorstore Files

Download the vectorstore files movies_faiss_index and recipes_faiss_index from the provided drive link in milestone3 and Load the following vectorstore files:
- `movies_faiss_index`
- `recipes_faiss_index`
- 
2. Change Folder Path in `llmmodel.py`

Navigate to `llmmodel.py` and update the folder path name to include the above downloaded folders

3. Run Streamlit Demo

Execute the stremlit_demo.py file using the following command on a Linux machine:

streamlit run streamlit_demo.py --server.fileWatcherType none
Afteer running the file, a link will be generate which we can access through the local machine to chat with the bot
