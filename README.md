# SMS Spam Classifier

Course project: classify an SMS as spam or ham (a normal message).

We train three models on the same split of a combined SMS collection, then explain the predictions with LIME and SHAP.

1. TF-IDF + logistic regression (baseline)
2. GRU in PyTorch (RNN)
3. Fine-tuned DistilBERT (transformer)

Spam is the positive class. About 61% of the combined messages are ham, so accuracy by itself is still a weak summary. We compare spam precision, recall, F1, and PR-AUC.

## Dataset

- Training file: `Code/sms_combined.csv`
- Original UCI file, kept as downloaded: `Code/sms_spam.csv`
- Sources: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and the [Super SMS Dataset](https://github.com/smspamresearch/spstudy)
- 67,709 unique messages: 41,390 ham (61.1%) and 26,319 spam (38.9%)
- Columns: `Category` (`ham` or `spam`) and `Text`

The Super SMS file was renamed to those two columns and merged with the UCI messages. A text that appears in both is stored once. Where the two sources disagreed (5 messages), we kept the UCI label. `sms+spam+collection.zip` is the original UCI download.

Please cite Almeida, Gómez Hidalgo, and Yamakami, "Contributions to the Study of SMS Spam Filtering," ACM DocEng 2011, and Salman, Ikram, and Kaafar, "Investigating Evasive Techniques in SMS Spam Filtering," IEEE Access 2024.

## Layout

```
Code/main.ipynb          training, plots, LIME, SHAP
Code/app_deploy.py       streamlit demo
Code/preprocess.py       text cleaning shared by the linear model, the GRU, and the app
Code/inference.py        loads the saved models for the app
Code/sms_spam.csv        original UCI messages
Code/sms_combined.csv    UCI plus Super SMS, the file the notebook loads
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
