from dataclasses import dataclass
from typing import Literal

from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import qdrant_client
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

st.set_page_config(page_title="Chat with the Chat Bot",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
os.environ["COHERE_API_KEY"] = st.secrets.cohere_key
QDRANT_HOST = st.secrets.QDRANT_HOST
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY

st.title("Welcome To GSM infoBot")


@dataclass
class Message:
  """Class for keeping track of chat messages."""
  origin: Literal["Customer", "elsa"]
  Message: str


def load_db():
  client = qdrant_client.QdrantClient(
      url=QDRANT_HOST,
      api_key=QDRANT_API_KEY,
  )
  embeddings = CohereEmbeddings(model="embed-english-v2.0")
  vector_store = Qdrant(client=client,
                        collection_name="my_documents",
                        embeddings=embeddings)
  return vector_store


def initialize_session_state():
  vector_store = load_db()
  # Initialize a session state to track whether the initial message has been sent
  if "initial_message_sent" not in st.session_state:
    st.session_state.initial_message_sent = False

  # Initialize a session state to store the input field value
  if "input_value" not in st.session_state:
    st.session_state.input_value = ""

  if "history" not in st.session_state:
    st.session_state.history = []

  if "chain" not in st.session_state:
    template = """
        Keep the response short \
        give response according to user question \
        the response should be under 15 words \
        response should be from the data source \
        respect the time of the user \
        try to fit in emojis possible in the response \
        Encourage users to visit the store without being pushy. \
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        {question}
        Answer:
        """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    memory = ConversationBufferWindowMemory(k=5,
                                            memory_key="history",
                                            input_key="question")
    chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    st.session_state.chain = RetrievalQA.from_chain_type(
        chat,
        chain_type='stuff',
        retriever=vector_store,
        chain_type_kwargs={
            "prompt": prompt,
            "memory": memory
        })


def on_click_callback():
  customer_prompt = st.session_state.customer_prompt

  if customer_prompt:
    st.session_state.input_value = ""
    st.session_state.initial_message_sent = True

    with st.spinner('Generating response...'):
      llm_response = st.session_state.run(
          {
              "context": st.session_state.chain.memory.buffer,
              "question": customer_prompt
          },
          return_only_outputs=True)

    st.session_state.history.append(Message("user", customer_prompt))
    st.session_state.history.append(Message("assistant", llm_response))


def main():
  initialize_session_state()
  chat_placeholder = st.container()
  prompt_placeholder = st.form("chat-form")

  with chat_placeholder:
    for chat in st.session_state.history:
      if type(chat.Message) is dict:
        msg = chat.Message['answer']
      else:
        msg = chat.Message
      st.write(msg)  # Print the message directly without any custom styling

  with st.form(key="chat_form"):
    cols = st.columns((6, 1))

    # Display the initial message if it hasn't been sent yet
    if not st.session_state.initial_message_sent:
      cols[0].text_input(
          "Chat",
          placeholder="Need Info? Ask Me Questions about GSM Mall's Features",
          label_visibility="collapsed",
          key="customer_prompt",
      )
    else:
      cols[0].text_input(
          "Chat",
          value=st.session_state.input_value,
          label_visibility="collapsed",
          key="customer_prompt",
      )

    cols[1].form_submit_button(
        "Ask",
        type="secondary",
        on_click=on_click_callback,
    )

  # Update the session state variable when the input field changes
  st.session_state.input_value = cols[0].text_input


if __name__ == "__main__":
  main()
