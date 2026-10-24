# SMS Spam Classifier

Course project: classify an SMS as spam or ham (a normal message).

We train three models on the same split of the UCI SMS Spam Collection, then explain the predictions with LIME and SHAP.

1. TF-IDF + logistic regression (baseline)
2. GRU in PyTorch (RNN)
3. Fine-tuned DistilBERT (transformer)

Spam is the positive class. About 87% of the messages are ham, so accuracy by itself is misleading. We compare spam precision, recall, F1, and PR-AUC.

## Dataset

- File: `Code/sms_spam.csv`
- Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- 5,574 messages: 4,827 ham (86.6%) and 747 spam (13.4%)
- Columns: `Category` (`ham` or `spam`) and `Text`

`sms+spam+collection.zip` is the original download. The notebook uses the csv. Duplicate texts are removed before the 80/10/10 split.

Please cite Almeida, Gómez Hidalgo, and Yamakami, "Contributions to the Study of SMS Spam Filtering," ACM DocEng 2011.

## Layout

```
Code/main.ipynb          training, plots, LIME, SHAP
Code/app_deploy.py       streamlit demo
Code/preprocess.py       text cleaning shared by the linear model, the GRU, and the app
Code/inference.py        loads the saved models for the app
Code/sms_spam.csv
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

1. Open `Code/main.ipynb` and run all cells. `RUN_RNN`, `RUN_BERT`, and `RUN_EXPLAIN` can be set to `False` if you want to skip a stage. The first DistilBERT run downloads the pretrained weights.

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

Add our names here.
