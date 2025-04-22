import streamlit as st
import requests

st.set_page_config(page_title = "arxivGPT", layout = "centered")
st.title("arxivGPT")

query = st.text_area("Ask a question about your PDFs : ", height = 100)

if st.button("Submit") and query.strip():
    with st.spinner("Thinking...")
    try : 
        response = requests.post("http://localhost:8000/chat", json={"question": query})
        st.markdown("### Response")
        st.write(response.json)
    except Exception as e:
        st.error(f"Something went wrong {e}")