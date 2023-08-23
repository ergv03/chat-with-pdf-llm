![image](https://github.com/ergv03/chat-with-pdf-llm/assets/23053920/a4160a7a-87de-43ef-a672-e946b1083840)

Simple web-based chat app, built using Streamlit.

Allows the user to talk with a LLM (today only OpenAI GPT is supported) and ask questions that can be answered by the PDFs provided by the user.

User needs to provide their own OpenAI API key.

## Instalation:

Just clone the repo and install the requirements using ```pip install -r requirements.txt```

## How to run locally:

Run ```streamlit run chat_app.py``` in your terminal.

Add the URLs of the PDF documents that are relevant to your queries, and start chatting with the bot. 

There's a level of context persistence, as bot has both conversational and document memories, but in order to keep the backend prompt within the model's token limit these memories follow a rolling window. In other words, the model is able to take into account only a small subset of messages that were exchanged (~4-5 most recent messages).
