# Code

This folder has the notebook and the small demo app.

| File | What it is |
| --- | --- |
| `sms_spam.csv` | UCI messages, columns `Category` and `Text` |
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
