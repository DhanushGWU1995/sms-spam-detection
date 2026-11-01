# Code

This folder has the notebook, the datasets, and the small demo app.

| File | What it is |
| --- | --- |
| `sms_spam.csv` | UCI SMS Spam Collection, columns `Category` (ham/spam) and `Text` |
| `super_sms_dataset.csv` | Super SMS Dataset, columns `SMSes` and `Labels` (1 = spam, 0 = ham) |
| `preprocess.py` | text cleaning used by logistic regression, the GRU, and the app |
| `main.ipynb` | combines both datasets, then EDA, the three models, LIME, and SHAP |
| `app_deploy.py` | Streamlit demo page |
| `inference.py` | loads the saved models for the app |
| `requirements.txt` | packages |
| `artifacts/` | train/val/test splits, metrics, and saved models from the notebook |

Run the notebook from this folder. Then, from the project root:

```bash
streamlit run Code/app_deploy.py
```
