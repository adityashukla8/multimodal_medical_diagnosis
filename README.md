# 🩺 MedFuse: A Multi-modal Clinical Case Retrieval Tool enabling diagnostic ease, triage readiness & medical staff training

MedFuse is an end-to-end pipeline that enables fast, accurate clinical case retrieval using multimodal data, combining both CT images and clinical text. It leverages Apache Beam (Dataflow), MongoDB Atlas vector search, and GCP services for scalable data processing.

---

## Architecture
<img width="3649" alt="Multi-modal_arch_1" src="https://github.com/user-attachments/assets/8fb5f36a-4547-4df4-9fb9-642bfbf30bcc" />

---

## What We're Solving

- **Healthcare data** is often large, complex, and scattered across formats (CSV metadata, case files in Parquet, and high-resolution medical images).
- Traditional tools struggle to process and scale with large image volumes, & manage multi-modal data (image + text).
- Clinicians need **quick, unified access** to patient data, history, and similar cases for **better decision-making, triage-assistance & staff education**
- Efficient storage, search, and retrieval of embeddings often becomes a bottleneck.
- We solve this by using **Google Cloud** for scalable processing, and **MongoDB Atlas** for efficient embedding storage, vector search, and retrieval.

- The pipeline processes heterogeneous patient data, extracts & stores embeddings,
  **enhancing medical workflows by enabling diagnostic ease, supporting triage decision-making through similar case retrieval, and accelerating training for medical staff.**

--- 

## How It Works

1. **Data Ingestion**  
   Medical images (e.g., CT, X-ray) and accompanying clinical text are ingested from a public dataset hosted on Google Cloud Storage.

2. **Embedding Generation via Apache Beam**  
   A scalable pipeline processes:
   - Visual features using a pretrained `DenseNet121` model (TensorFlow)
   - Textual features using `Bio_ClinicalBERT` (HuggingFace Transformers)  
   These embeddings are combined and stored in a MongoDB Atlas collection.

3. **Vector Indexing & Search**  
   A MongoDB Atlas vector index enables efficient retrieval of similar cases by computing similarity over the combined feature space.

4. **Interactive Web App (Streamlit)**  
   Users can:
   - Search by **clinical text**, **image**, or both
   - Upload images or use sample GCS links
     > Image upload support coming soon!
   - View retrieved similar cases along with findings, captions, and scores

---

## Examples

Try the following to see case retrieval in action hosted at https://multimodal-medical-diagnosis-502131642989.us-central1.run.app/Find_Similar_Cases:

### Example 1: Head Trauma Case
> Unconscious patient following a fall. CT scan shows crescent-shaped hyperdensity along left hemisphere with midline shift. Likely subdural hematoma.

### Example 2: Text + Image

Use the following clinical text and GCS image path together:

**Clinical Text:**
> non-contrast head ct (transverse views) shows chronic right anterior cerebral artery (aca) and middle cerebral artery (mca) territory encephalomalacia with sparing of a small portion of the medial right frontal lobe, without evidence of bleed or new territory of acute infarction.
**Image GCS Path:**
```gs://multicare-dataset/pmc4/pmc481/pmc4815137_fneur-07-00050-g001_a_1_2.jpg```

---

## Dataset

- Source: MultiCaRe - A Multi-Modal Clinical Dataset
- Contains 75,000+ open-access case report articles
  - ~96,400 individual clinical cases
  - ~135,600 medical images (CT, MRI, X‑ray, pathology)
- Covers multiple specialties: oncology, cardiology, surgery, pathology
- Includes rich metadata: article info, patient info, image captions & labels, free‑text case descriptions

---

## Tech Stack

- Python, PyTorch, TensorFlow, Hugging Face Transformers
- Apache Beam, Google Cloud Dataflow
- MongoDB Atlas Vector Search
