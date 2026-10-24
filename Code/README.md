# Code

Training notebook and demo app for SMS spam detection.

| File | Role |
| --- | --- |
| `sms_spam.csv` | UCI SMS Spam Collection as `Category`, `Text` |
| `preprocess.py` | Cleaning shared by the classical model, the GRU, and the app |
| `main.ipynb` | EDA, TF-IDF + logistic regression, GRU, DistilBERT, LIME, and SHAP |
| `app_deploy.py` | Streamlit demo that scores one SMS with logistic regression, the GRU, and DistilBERT |
| `inference.py` | Loads those three saved models for the demo |
| `requirements.txt` | Python dependencies |
| `artifacts/` | Created by the notebook: split files, models, and metrics |

Run the notebook from this directory, then start the app from the repository root:

```bash
streamlit run Code/app_deploy.py
```
