# MedFuse: A Multi-modal Clinical Case Retrieval Tool enabling diagnostic ease, triage readiness & medical staff training

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
