import streamlit as st

st.set_page_config(page_title="Report Generator", layout="wide")

st.title("📋 VLM-assisted Report Generator")
st.write("Generate structured diagnostic reports using visual language models (e.g., MedGemma).")

image_path = st.text_input("Enter GCS path to the image (gs://...)")

if st.button("Generate Report"):
    # Placeholder — you can plug in MedGemma call here
    st.success("✅ Report generated!")
    st.markdown("**Sample Report:**")
    st.markdown("""
    **Findings**: There is mild ground-glass opacity in the lower lobes.
    
    **Impression**: Possible early pneumonia or COVID-19. Recommend follow-up.
    
    **Next Steps**: RT-PCR, clinical correlation, repeat CT in 5-7 days.
    """)
