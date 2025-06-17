import streamlit as st

st.set_page_config(page_title="MedFuse")

st.title("🩺 MedFuse")
st.markdown("#### *A Multi-modal Clinical Case Retrieval System*")
st.markdown("##### *Enabling diagnostic ease, triage readiness and medical staff training*")

st.subheader("Get Started:")

# col1, col2 = st.columns(2)

# with col1:
if st.button("🔍 Find Similar Cases", use_container_width=True):
    st.switch_page("pages/2_Find_Similar_Cases.py")

st.header("About MedFuse")
st.markdown("""MedFuse enhances medical workflows by enabling diagnostic ease, supporting triage decision-making through similar case retrieval, and accelerating training for medical staff.
            
#### What we're Solving:
- Clinicians often need ***quick, unified*** access to patient data, history, and similar cases for ***better decision-making, triage-assistance & staff training***
- Traditional tools struggle to process and scale large image volumes, & managing multi-modal data (image + text).
- Efficient storage, search, and retrieval of embeddings often becomes a bottleneck.
            
#### MedFuse Features:
- **Multi-modal search** over CT images + clinical notes
- **Real-time case retrieval** using MongoDB vector search
""")