# FairFrame (STEMagine)

> Aligning generative AI to societal values: detect representation gaps in image-generation prompts and rewrite them to be more inclusive.

Prompts like *"Generate a picture of a researcher"* tend to produce the same stereotypical image every time. **FairFrame** analyzes a prompt, detects which demographic dimensions are left unspecified (gender, race, age, disability), and suggests a rewritten, more diverse prompt, e.g. *"Generate a picture of a Black female researcher in a wheelchair."*

## How it works

```
User prompt ──► Detector (DeBERTa-v3, multi-label) ──► gaps found? ──► Rewriter (Llama-3 8B) ──► inclusive prompt
                                                         └── no ──► prompt is already safe
```

1. **Detector:** a `microsoft/deberta-v3-base` model fine-tuned for multi-label classification (`Safe`, `Gender`, `Race`, `Age`, `Disability`, `Profession`) using sigmoid outputs, so each gap is predicted independently.
2. **Rewriter:** Llama-3 8B Instruct (4-bit) rewrites the prompt, filling the detected gaps with attributes sampled from `demographics.json`.
3. **Web app:** a Flask front end (`app.py`, `templates/`, `static/`) that calls both services and shows the detected gaps and the rewritten prompt.

All model services are deployed as serverless GPU endpoints on [Modal](https://modal.com).

## Data

- `generatedDataset.csv`: synthetic prompt pairs (underspecified vs. inclusive) with multi-hot gap labels, built with `src/createDataset.py`.
- `cpDataset.csv`: the [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) social-bias benchmark, used as additional training data.

## Project structure

```
├── app.py                  # Flask web app (deployed on Modal)
├── demographics.json       # attribute options used by the rewriter
├── generatedDataset.csv    # synthetic training data
├── cpDataset.csv           # CrowS-Pairs bias dataset
├── src/
│   ├── createDataset.py    # builds the synthetic dataset
│   ├── trainClassifier.py  # fine-tunes the DeBERTa multi-label detector
│   ├── tesClassifier.py    # runs test prompts through the detector
│   ├── trainLLM.py         # Llama-3 based classifier service
│   └── reWriter.py         # Llama-3 prompt rewriter service
├── templates/index.html    # UI
└── static/style.css
```

## Getting started

```bash
pip install modal flask requests
modal setup                              # authenticate with Modal

modal run src/trainClassifier.py         # train the detector
modal run src/tesClassifier.py           # sanity-check it on test prompts
modal deploy src/reWriter.py             # deploy the rewriter
modal deploy app.py                      # deploy the web app
```

Update `DETECTOR_URL` and `REWRITER_URL` in `app.py` to point to your own deployed endpoints.

## Tech stack

Python · PyTorch · Hugging Face Transformers · DeBERTa-v3 · Llama-3 · Flask · Modal
