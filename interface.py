import streamlit as st
from agents import Agent

agent = Agent()

st.title("RAG + Multi-Agent Q&A Assistant")
query = st.text_input("Ask your question:")

if query:
    response = agent.run(query)
    st.write("### Answer:")
    st.write(response["answer"])

    st.write("---")
    st.write("### Agent Used:")
    st.write(response["used_tool"])

    st.write("### Retrieved Context:")
    st.write(response["context"])
