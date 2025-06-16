import streamlit as st

st.set_page_config(page_title="MedFuse")

st.title("🩺 MedFuse")
st.markdown("#### *A Multi-modal Clinical Case Retrieval & Report Generation Tool*")

st.write("Choose an option to get started:")

col1, col2 = st.columns(2)

# with col1:
if st.button("🔍 Find Similar Cases", use_container_width=True):
    st.switch_page("pages/2_Find_Similar_Cases.py")

# with col2:
if st.button("📝 Generate Clinical Report", use_container_width=True):
    st.switch_page("pages/3_Generate_Reports.py")


st.markdown("""
#### What we're Solving:
- Clinicians often need ***quick, unified*** access to patient data, history, and similar cases for ***better decision-making, triage-assistance & staff education***
- Traditional tools struggle to process and scale large image volumes, & managing multi-modal data (image + text).
- Efficient storage, search, and retrieval of embeddings often becomes a bottleneck.
            
#### MedFuse Features:
- **Multi-modal search** over CT images + clinical notes
- **Real-time case retrieval** using MongoDB vector search
- **Report generation** powered by VLMs like MedGemma""")

