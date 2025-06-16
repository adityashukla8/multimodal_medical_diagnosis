# MedFuse: A Multi-modal Clinical Case Retrieval & Report Generation Tool

MedFuse is an end-to-end pipeline that enables fast, accurate clinical case retrieval using multimodal data — combining both CT images and clinical text. It leverages Apache Beam (Dataflow), MongoDB Atlas vector search, and GCP services for scalable data processing.

---

## Architecture
<img width="4324" alt="Multi-modal_arch" src="https://github.com/user-attachments/assets/bc382a8e-1f2d-47c0-b41b-3070614334b0" />

---

## What We're Solving

- **Healthcare data** is often large, complex, and scattered across formats (CSV metadata, case files in Parquet, and high-resolution medical images).
- Traditional tools struggle to process and scale with large image volumes, & managing multi-modal data (image + text).
- Clinicians need **quick, unified access** to patient data, history, and similar cases for **better decision-making, triage-assistance & staff education**
- Efficient storage, search, and retrieval of embeddings often becomes a bottleneck.
- We solve this by using **Google Cloud** for scalable processing, and **MongoDB Atlas** for efficient embedding storage, vector search, and retrieval.

- The pipeline processes patient data, extracts & stores embeddings, integrates GenAI,
  **enabling multi-modal case search and report generation via VLM (e.g., MedGemma) for downstream clinical workflows.**

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
