import streamlit as st
import os
from constants import search_number_messages
from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_pdf
import re


def remove_url(url_to_remove):
    """
    Remove URLs from the session_state. Triggered by the respective button
    """
    if url_to_remove in st.session_state.urls:
        st.session_state.urls.remove(url_to_remove)


# Page title
st.set_page_config(page_title='Talk with PDFs using LLMs - Beta')
st.title('Talk with PDFs using LLMs - (Beta)')

# Initialize the faiss_index key in the session state. This can be used to avoid having to download and embed the same PDF
# every time the user asks a question
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = {
        'indexed_urls': [],
        'index': None
    }

# Initialize conversation memory used by Langchain
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = None

# Initialize chat history used by StreamLit (for display purposes)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the URLs added by the user in the UI
if 'urls' not in st.session_state:
    st.session_state.urls = []

with st.sidebar:

    openai_api_key = st.text_input('Step 1 - OpenAI API Key:', type='password')

    # Add/Remove URLs form
    with st.form('urls-form', clear_on_submit=True):
        url = st.text_input('Step 2 - URLs to relevant PDFs: ')
        add_url_button = st.form_submit_button('Add')
        if add_url_button:
            if url not in st.session_state.urls:
                st.session_state.urls.append(url)

    # Display a container with the URLs added by the user so far
    with st.container():
        if st.session_state.urls:
            st.header('URLs added:')
            for url in st.session_state.urls:
                st.write(url)
                st.button(label='Remove', key=f"Remove {url}", on_click=remove_url, kwargs={'url_to_remove': url})
                st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query_text := st.chat_input("Your message"):

    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Display user message in chat message container, and append to session state
    st.chat_message("user").markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Check if FAISS index already exists, or if it needs to be created as it includes new URLs
    session_urls = st.session_state.urls
    if st.session_state['faiss_index']['index'] is None or set(st.session_state['faiss_index']['indexed_urls']) != set(session_urls):
        st.session_state['faiss_index']['indexed_urls'] = session_urls
        with st.spinner('Downloading and indexing PDFs...'):
            faiss_index = download_and_index_pdf(session_urls)
            st.session_state['faiss_index']['index'] = faiss_index
    else:
        faiss_index = st.session_state['faiss_index']['index']

    # Check if conversation memory has already been initialized and is part of the session state
    if st.session_state['conversation_memory'] is None:
        conversation = initialize_chat_conversation(faiss_index)
        st.session_state['conversation_memory'] = conversation
    else:
        conversation = st.session_state['conversation_memory']

    # Search PDF snippets using the last few user messages
    user_messages_history = [message['content'] for message in st.session_state.messages[-search_number_messages:] if message['role'] == 'user']
    user_messages_history = '\n'.join(user_messages_history)

    with st.spinner('Querying OpenAI GPT...'):
        response = conversation.predict(input=query_text, user_messages_history=user_messages_history)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from page {page_number + 1}'):
                # Remove the <START> and <END> tags from the snippets before displaying them
                snippet = re.sub("<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\d+>", '', snippet)
                st.markdown(snippet)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
