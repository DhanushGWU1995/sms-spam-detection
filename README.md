# SMS Spam Classifier

This is our NLP final project. The goal is to classify a text message as spam or ham (a normal message).

We trained three models on the same data and compared them:

1. TF-IDF + logistic regression (our baseline)
2. GRU in PyTorch (our RNN)
3. Fine-tuned DistilBERT (our transformer)

Then we used LIME and SHAP to see which words each model was actually paying attention to.

Spam is the class we care about. Around 61% of the messages are ham and 39% are spam, so accuracy alone can look good even when a lot of spam gets through. That's why we focus on spam precision, recall, F1, and PR-AUC.

## Dataset

We combined two datasets:

| Dataset | File | Ham | Spam | Total |
|---|---|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | `Code/sms_spam.csv` | 4,827 | 747 | 5,574 |
| [Super SMS Dataset](https://github.com/smspamresearch/spstudy) (2012–2024) | `Code/super_sms_dataset.csv` | 40,830 | 26,178 | 67,008 |
| **Combined** | | **45,657** | **26,925** | **72,582** |

The two files have different column names, so the notebook puts them in the same format (`Text`, `label`, and a `Source` column so we know where each message came from). Some messages show up in both datasets, so we removed duplicates before splitting. That took out 4,340 rows plus 1 message that was labeled both ham and spam. We ended up with **68,242 messages** (61.3% ham, 38.7% spam) and split them 80/10/10 into train, validation, and test.

Datasets we used:
- T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "Contributions to the Study of SMS Spam Filtering: New Collection and Results," ACM DocEng, 2011.
- M. Salman, M. Ikram, and M. A. Kaafar, "Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models," IEEE Access, 2024. https://ieeexplore.ieee.org/document/10431737

## Results

All three models were tested on the same 6,825 test messages:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.990 | 0.990 | 0.984 | 0.987 | 0.998 |
| GRU | 0.988 | 0.983 | 0.984 | 0.984 | 0.998 |
| **DistilBERT** | **0.994** | **0.993** | **0.992** | **0.992** | **0.999** |

DistilBERT did the best on every metric and missed about half as much spam as the other two. Logistic regression was surprisingly close for a much simpler model. We ran everything on an AWS g5.xlarge GPU. The full numbers are in `Code/artifacts/metrics.csv`.

## What's in the repo

```
Code/main.ipynb             the whole pipeline: EDA, training, results, LIME, SHAP
Code/app_deploy.py          Streamlit demo app
Code/preprocess.py          text cleaning used by logistic regression, the GRU, and the app
Code/inference.py           loads the saved models for the app
Code/sms_spam.csv           UCI dataset
Code/super_sms_dataset.csv  Super SMS dataset
Code/artifacts/             saved models and results (created when you run the notebook)
Group-Proposal/
Final-Group-Presentation/
Final-Group-Project-Report/
```

## Setup

You need Python 3.10 or newer.

```bash
pip install -r Code/requirements.txt
```

## How to run it

1. Open `Code/main.ipynb` and run all the cells. If you want to skip a part, set `RUN_RNN`, `RUN_BERT`, or `RUN_EXPLAIN` to `False` at the top. DistilBERT downloads its pretrained weights the first time. The dataset is pretty big, so use a GPU for DistilBERT or set `BERT_EPOCHS = 1`.

   The notebook saves:

   - `Code/artifacts/vectorizer.pkl`
   - `Code/artifacts/logreg_model.pkl`
   - `Code/artifacts/threshold.json`
   - `Code/artifacts/gru.pt`
   - `Code/artifacts/distilbert/`
   - `Code/artifacts/metrics.csv`

2. Then, from the project folder, start the app:

```bash
streamlit run Code/app_deploy.py
```

Paste in any text message and the app shows what logistic regression, the GRU, and DistilBERT each think. A model only shows up if you've run its part of the notebook.

## What we report

For each model, on the same test set:

- accuracy
- spam precision, recall, and F1
- PR-AUC
- confusion matrix

Each model picks its spam cutoff on the validation set, and then we use that cutoff on the test set.

## Team

Alex and Dhanush
