import streamlit as st
from model_invoker import orchestrator

# Title Displayed on the Streamlit Web App
st.title(f""":rainbow[Amazon Bedrock Gen AI Model Playground]""")

# Configuring Values for Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Putting a Sidebar in with a Select Box to select the LLM you want to use
model_id = st.sidebar.selectbox("Select A Model", [
    'anthropic.claude-instant-v1',
    'anthropic.claude-v2',
    'anthropic.claude-v2:1',
    'antrhopic.claude-3-haiku-20240307-v1:0',
    'antrhopic.claude-3-sonnet-20240229-v1:0',
    'mistral.mistral-7b-instruct-v0:2',
    'mistral.mixtral-8x7b-instruct-v0:1',
    'mistral.mistral-large-2402-v1:0',
    'meta.llama2-13b-chat-v1',
    'meta.llama2-70b-chat-v1',
    'meta.llama3-8b-instruct-v1:0',
    'meta.llama3-70b-instruct-v1:0',
    'cohere.command-text-v14',
    'cohere.command-light-text-v14',
    'amazon.titan-text-lite-v1',
    'amazon.titan-text-express-v1',
    'ai21.j2-mid-v1',
    'ai21.j2-ultra-v1'
    ])

# Evaluating st.chat_input and determining if a question has been input
if question := st.chat_input("Ask me about anything...but actually...anything..."):
    # With the user icon, write the question to the front end
    with st.chat_message("user"):
        # Writing the question to the front end
        st.markdown(question)
    # Append the question and the role (user) as a message to the session state
    st.session_state.messages.append({"role": "user",
                                      "content": question})
    # Respond as the assisntant with the answer
    with st.chat_message("assistant"):
        # Making sure there are no messages present when generating the answer
        message_placeholder = st.empty()
        # Putting a spinning icon to show that the query is in progress
        with st.status("Determining the best possible answer!", expanded=False) as status:
            # Passing the question into the orchestrator, which then invokes the approprate LLM
            answer = orchestrator(question, model_id)
            # Writing the answer to the front end
            message_placeholder.markdown(f"{answer}")
            # Showing a completion message to the front end
            status.update(label="Question Answered...", state="complete", expanded=False)
    # Appending the results to the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})
