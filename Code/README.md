# Code

This folder has the notebook and the small demo app.

| File | What it is |
| --- | --- |
| `super_sms_dataset.csv` | Super SMS Dataset, columns `SMSes` and `Labels` (1 = spam, 0 = ham) |
| `preprocess.py` | cleaning used by logistic regression, the GRU, and the app |
| `main.ipynb` | EDA, the three models, LIME, and SHAP |
| `app_deploy.py` | Streamlit page |
| `inference.py` | loads the saved models for the app |
| `requirements.txt` | packages |
| `artifacts/` | split, metrics, and model files from the notebook |

Run the notebook from this folder. Then, from the project root:

```bash
streamlit run Code/app_deploy.py
```
