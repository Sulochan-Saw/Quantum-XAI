# Quantum XAI — Member 7 Contribution Pack
**Contributor:** Krishnangshubhanja  
**Role:** Application interface, project README, presentation, and report formatting  
**Project theme:** Explainable hybrid quantum-classical learning for cancer drug-sensitivity prediction.

## What is included
- `app/app.py`: Streamlit demo dashboard. It accepts a tabular CSV, trains a classical Random Forest baseline, and displays metrics and feature importance.
- `app/quantum_circuit_template.py`: PennyLane variational-circuit template for the team to connect to its verified quantum model.
- `data/demo_drug_sensitivity.csv`: Small synthetic teaching dataset, not biological or clinical data.
- `docs/Project_Report_Member7.docx`: formatted report draft for the application/documentation contribution.
- `docs/Project_Presentation_Member7.pptx`: editable presentation deck.
- `requirements.txt`: Python dependencies.
- `MANUAL_GITHUB_UPLOAD.md`: step-by-step instructions to add this contribution to the existing repository.

## Important scientific limitations
This is a **prototype and documentation contribution**, not a validated medical tool. The included CSV is synthetic. The dashboard's Random Forest is a classical baseline only. The PennyLane file is a circuit template and does not claim a trained quantum model, quantum advantage, or verified clinical predictions. Replace the synthetic data and connect the team's actual preprocessing, quantum model, and explainability outputs before reporting project results.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Expected CSV format
A header row, one row per sample, numeric feature columns, and one target column (for example `sensitivity_class`). Choose the target column in the dashboard. Do not upload identifiable patient data.
