import streamlit as st
from knowledge_base import KnowledgeBaseService

# Page title
st.title("Knowledge Base Update Service")


@st.cache_resource
def get_knowledge_service():
    """Cache the knowledge base service so Chroma and the splitter are not rebuilt on every upload."""
    return KnowledgeBaseService()


# file_uploader
uploader_file = st.file_uploader(
    label="Please upload a TXT file",
    type=['txt'],
    accept_multiple_files=False,   # False means only a single file can be uploaded
)

if uploader_file is not None:
    # Extract file information
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024      # KB

    st.subheader(f"File name: {file_name}")
    st.write(f"Type: {file_type} | Size: {file_size:.2f} KB")

    # get_value -> bytes -> decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8")

    with st.expander("View file content"):
        st.write(text)

    if st.button("Load into knowledge base"):
        with st.spinner("Vectorizing and writing to the knowledge base..."):
            service = get_knowledge_service()
            result = service.upload_by_str(text, file_name)
        st.success(result)
