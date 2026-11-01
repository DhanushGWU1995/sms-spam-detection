# SMS Spam Classifier

Course project: classify an SMS as spam or ham (a normal message).

We train three models on the same split of the Super SMS Dataset, then explain the predictions with LIME and SHAP.

1. TF-IDF + logistic regression (baseline)
2. GRU in PyTorch (RNN)
3. Fine-tuned DistilBERT (transformer)

Spam is the positive class. About 61% of the messages are ham and 39% are spam, so accuracy by itself can hide missed spam. We compare spam precision, recall, F1, and PR-AUC.

## Dataset

- File: `Code/super_sms_dataset.csv`
- Source: [Super SMS Dataset](https://github.com/smspamresearch/spstudy)
- About 68,000 messages collected from 2012 to 2024, roughly 61% ham and 39% spam
- Columns: `SMSes` (the message) and `Labels` (`1` = spam, `0` = ham)

The notebook renames the columns to `Text` and `label`. Texts that appear with both labels are dropped, then duplicate texts are removed before the 80/10/10 split.

Please cite M. Salman, M. Ikram, and M. A. Kaafar, "Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models," IEEE Access, 2024. https://ieeexplore.ieee.org/document/10431737

## Layout

```
Code/main.ipynb          training, plots, LIME, SHAP
Code/app_deploy.py       streamlit demo
Code/preprocess.py       text cleaning shared by the linear model, the GRU, and the app
Code/inference.py        loads the saved models for the app
Code/super_sms_dataset.csv
Code/artifacts/          created when the notebook is run
Group-Proposal/
Final-Group-Presentation/
Final-Group-Project-Report/
```

## Setup

Python 3.10 or newer.

```bash
pip install -r Code/requirements.txt
```

## How to run

1. Open `Code/main.ipynb` and run all cells. `RUN_RNN`, `RUN_BERT`, and `RUN_EXPLAIN` can be set to `False` if you want to skip a stage. The first DistilBERT run downloads the pretrained weights. The dataset is large, so use a GPU for DistilBERT or set `BERT_EPOCHS = 1`.

   The notebook writes:

   - `Code/artifacts/vectorizer.pkl`
   - `Code/artifacts/logreg_model.pkl`
   - `Code/artifacts/threshold.json`
   - `Code/artifacts/gru.pt`
   - `Code/artifacts/distilbert/`
   - `Code/artifacts/metrics.csv`

2. From the project folder:

```bash
streamlit run Code/app_deploy.py
```

Paste a message and the app shows the logistic regression, GRU, and DistilBERT decisions next to each other. A model only appears if that notebook section has been run and the file is in `artifacts/`.

## What we report

For every model, on the same test set:

- accuracy
- spam precision, recall, and F1
- PR-AUC
- confusion matrix

The spam threshold is chosen on the validation set, then applied to the test set.

## Team

Alex and Dhanush
