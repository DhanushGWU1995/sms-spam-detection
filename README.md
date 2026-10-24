# SMS Spam Classifier

Binary text classifier that labels an SMS as **spam** or **ham** (legitimate). The project is organized like a course repository: one training notebook, a Streamlit demo, and separate folders for the proposal, the final report, and the presentation.

## Project overview

Spam texts are short, informal, and often obfuscated (`fr33`, short codes, currency amounts). The notebook trains three models on the same split so the comparison is fair:

1. **TF-IDF + logistic regression** — classical baseline
2. **GRU in PyTorch** — recurrent network
3. **Fine-tuned DistilBERT** — transformer
4. **LIME and SHAP** — local and global explanations on a shared set of messages

Spam is the positive class. The headline metrics are spam precision, recall, F1, and PR-AUC. Accuracy is reported too, and it is a weak summary on this dataset because about 87% of messages are ham.

## Dataset

- **File**: `Code/sms_spam.csv`
- **Source**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Size**: 5,574 messages — 4,827 ham (86.6%) and 747 spam (13.4%)
- **Columns**:
  - `Category`: `ham` or `spam`
  - `Text`: raw message

The original download is `sms+spam+collection.zip` in the repository root. `Code/sms_spam.csv` is the working copy. The notebook drops duplicate message text before the stratified 80/10/10 split.

Cite the collection as Almeida, Gómez Hidalgo, and Yamakami, “Contributions to the Study of SMS Spam Filtering,” ACM Symposium on Document Engineering, 2011.

## Repository layout

```
README.md
.devcontainer/devcontainer.json
sms+spam+collection.zip
Code/
  main.ipynb
  app_deploy.py
  preprocess.py
  requirements.txt
  sms_spam.csv
  artifacts/            # models and metrics, created by the notebook
Group-Proposal/
Final-Group-Presentation/
Final-Group-Project-Report/
```

## Requirements

- Python 3.10+
- Packages in `Code/requirements.txt`

```bash
pip install -r Code/requirements.txt
```

## How to run

1. Train and evaluate from the `Code` directory by opening `Code/main.ipynb` and running all cells. The notebook writes:

   - `Code/artifacts/vectorizer.pkl`
   - `Code/artifacts/logreg_model.pkl`
   - `Code/artifacts/threshold.json`
   - `Code/artifacts/gru.pt`
   - `Code/artifacts/distilbert/`
   - `Code/artifacts/metrics.csv`

2. Start the demo from the repository root:

```bash
streamlit run Code/app_deploy.py
```

The app classifies a pasted SMS with the logistic regression model. Run the notebook once so those artifact files exist.

`RUN_RNN`, `RUN_BERT`, and `RUN_EXPLAIN` at the top of the modeling section can be set to `False` to skip a stage.

## Evaluation

Each model is scored on the same held-out test set:

- Accuracy
- Spam precision, recall, and F1
- PR-AUC
- Confusion matrix

The spam decision threshold is chosen on the validation split, then applied once to the test split.

## Contributors

Add the team names for the course submission here.
