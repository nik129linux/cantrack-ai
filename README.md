# CanTrack AI

Installable Python package `cantrack_ai`, with CLIP image embeddings, image validation,
cosine similarity and vision helpers. Runtime source is under `src/cantrack_ai`.
The backend consumes this repository as a locked Git dependency.
The current Gemini/Groq provider chain and walk photo-condition orchestration live in
`cantrack-backend/apps/api-py/src/cantrack_api/ai_chain.py` and `walk_photo_ai.py`.

## Installation and tests

Use Python 3.12 or newer:

```sh
python -m pip install -e . pytest onnxruntime
python -m pytest tests/py/ai -q
```

CLIP inference needs ONNX Runtime and model weights. The API Dockerfile downloads the
model at build time; installing this package alone does not pre-download it.
Tests substitute model/provider dependencies and are not a measurement of real model accuracy.
See `evals/clip_eval.py` for the separate evaluation tool.
Never commit provider credentials or private user photos.

## Project repositories

| Component | Repository |
|---|---|
| Frontend | https://github.com/nik129linux/cantrack-frontend |
| Backend | https://github.com/nik129linux/cantrack-backend |
| Database | https://github.com/nik129linux/cantrack-db |
| AI package | https://github.com/nik129linux/cantrack-ai |
