import time
from rag import RagService
import streamlit as st
import config_data as config

# Title
st.title("AI Customer Service")
st.divider()            # Divider

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "Hello, how can I help you?"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# Provide an input box at the bottom of the page
prompt = st.chat_input()

if prompt:

    # Show the user's question on the page
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("AI is thinking..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        # yield

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

# ["a", "b", "c"]   "".join(list)    -> abc
# ["a", "b", "c"]   ",".join(list)    -> a,b,c
