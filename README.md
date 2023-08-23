
![image](https://github.com/ergv03/chat-with-pdf-llm/assets/23053920/969edf03-4451-4909-98d9-601d92a17e83)

## Overview:

Simple web-based chat app, built using [Streamlit](https://streamlit.io/) and [Langchain](https://python.langchain.com/). The app backend follows the Retrieval Augmented Generation (RAG) framework.

Allows the user to provide a list of PDFs, and ask questions to a LLM (today only OpenAI GPT is implemented) that can be answered by these PDF documents.

User needs to provide their own OpenAI API key.

## Instalation:

Just clone the repo and install the requirements using ```pip install -r requirements.txt```

## How to run locally:

Run ```streamlit run chat_app.py``` in your terminal.

Add the URLs of the PDF documents that are relevant to your queries, and start chatting with the bot. 

## How it works:

The provided PDFs will be downloaded and properly split into chunks, and finally embedding vectors for each chunk will be generated using OpenAI service. These vectors are then indexed using FAISS, and can be quickly retrieved.

As the user interacts with the bot, new relevant document chunks/snippets are retrieved and added to the session memory, alongside the past few messages. These snippets and messages are part of the prompt sent to the LLM; this way, the model will have as context not just the latest message and retrieved snippet, but past ones as well.
