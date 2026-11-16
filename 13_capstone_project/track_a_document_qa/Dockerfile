# Track A image. BUILD CONTEXT = THE REPO ROOT (the service imports modules 14, 18, 19, 20 by path, it does not copy them):
#   docker build -f 13_capstone_project/track_a_document_qa/Dockerfile -t docqa .        (run from the repo root)
# NOT built or run in this course (no Docker daemon assumed); read it as a reviewed recipe, see README "Docker".
FROM python:3.12-slim
WORKDIR /repo

# 1. dependencies first: this layer is cached until requirements.txt changes. CPU-only torch keeps the image far smaller than the CUDA wheel.
COPY 13_capstone_project/track_a_document_qa/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# 2. the two small embedding / rerank models are baked into the image at BUILD time (needs network once), so the running container is offline
ENV HF_HOME=/opt/hf
COPY 13_capstone_project/track_a_document_qa/prefetch_models.py /tmp/prefetch_models.py
RUN python /tmp/prefetch_models.py
ENV HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# 3. only the code the pipeline imports, keeping the repo-relative paths docqa.py computes (REPO = parents[1] of the track folder)
COPY 14_rag_and_retrieval/common.py                      14_rag_and_retrieval/common.py
COPY 18_ocr_and_document_parsing/docimg.py               18_ocr_and_document_parsing/docimg.py
COPY 18_ocr_and_document_parsing/dococr.py               18_ocr_and_document_parsing/dococr.py
COPY 18_ocr_and_document_parsing/07_reading_order_and_tables/main.py     18_ocr_and_document_parsing/07_reading_order_and_tables/main.py
COPY 18_ocr_and_document_parsing/08_pdf_internals_and_routing/          18_ocr_and_document_parsing/08_pdf_internals_and_routing/
COPY 18_ocr_and_document_parsing/10_parsed_document_handoff/main.py      18_ocr_and_document_parsing/10_parsed_document_handoff/main.py
COPY 19_chunking_deep_dive/common19.py                   19_chunking_deep_dive/common19.py
COPY 19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/    19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/
COPY 20_llm_system_lld/ragkit/                           20_llm_system_lld/ragkit/
COPY 13_capstone_project/track_a_document_qa/            13_capstone_project/track_a_document_qa/

# 4. run as a non-root user (same pattern as the core capstone api/Dockerfile); writable dirs for artifacts
RUN useradd --create-home appuser && chown -R appuser /repo
USER appuser
WORKDIR /repo/13_capstone_project/track_a_document_qa
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8001/health')"
# 0.0.0.0 so the published port works; ONE worker because each worker would load its own copy of the index and models
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
