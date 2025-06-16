import streamlit as st

st.set_page_config(page_title="MedFuse")

st.title("🧠 MedFuse")
st.subheader("A Multi-modal Clinical Case Retrieval & Report Generation Tool")

st.markdown("#### Choose an option to get started:")

col1, col2 = st.columns(2)

# with col1:
if st.button("🔍 Find Similar Cases", use_container_width=True):
    st.switch_page("pages/2_Case_Search.py")

# with col2:
if st.button("📝 Generate Report", use_container_width=True):
    st.switch_page("pages/2_Report_Generation.py")


st.markdown("""
            
### What MedFuse Offers
- **Multi-modal search** over CT images + clinical notes
- **Real-time case retrieval** using MongoDB vector search
- **Report generation** powered by VLMs like MedGemma""")