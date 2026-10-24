# Project Proposal: SMS Spam Detection

## Team

Add each team member's name and role before submission.

## Problem

Phone users still receive unsolicited SMS: prize scams, premium-rate traps, and phishing. Hand-written keyword rules miss obfuscated wording and also block ordinary messages that happen to contain words like "free" or "call". This project builds a supervised classifier that reads the message text and labels it spam or ham, then explains which tokens drove the decision.

## Dataset

The [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) contains 5,574 English SMS messages labeled ham or spam (4,827 ham, 747 spam). Messages are short, informal, and full of abbreviations. Duplicate texts will be removed before splitting so the same message cannot appear in both train and test. The remaining rows will be split 80/10/10, stratified on the label, with one fixed seed shared by every model.

Spam is the minority class (13.4%). A classifier that always predicts ham is already about 87% accurate, so model selection will use spam precision, recall, F1, and precision-recall AUC.

## Proposed approach

One task, three model families, and one explanation layer.

| Stage | Method | What it covers |
| --- | --- | --- |
| Classical baseline | Word and character TF-IDF, then L2 logistic regression with balanced class weights | Bag-of-n-grams features and a linear classifier |
| Recurrent network | GRU over learned token embeddings, trained in PyTorch with a class-weighted loss | Sequence modeling |
| Transformer | Fine-tuned `distilbert-base-uncased` (max length 128, a few epochs) | Contextual representations |
| Interpretability | Coefficient lists, LIME on individual messages, and SHAP on the linear model | Why a message was labeled spam |

Cleaning stays light. Digits, currency marks, and short codes are spam evidence, so they are kept for the classical model and the GRU. DistilBERT reads the raw message through its own tokenizer.

The spam threshold for each model is selected on the validation set by spam F1. Test labels are used only for the final table.

## Demo

`Code/app_deploy.py` is a Streamlit app. After the notebook saves all three models, the app scores one pasted SMS with logistic regression, the GRU, and DistilBERT, and shows each spam probability.

## Deliverables

| Folder | Contents |
| --- | --- |
| `Code/` | Notebook, demo, dataset, and saved models |
| `Group-Proposal/` | This proposal |
| `Final-Group-Presentation/` | Slide deck |
| `Final-Group-Project-Report/` | Written report with the comparison table, plots, and error analysis |

## Evaluation plan

Report, for every model:

- Accuracy
- Spam precision, recall, and F1
- PR-AUC
- Confusion matrix

Then a short error analysis of false negatives (missed spam) and false positives (blocked ham), using the same example messages for LIME and SHAP.

## References

- Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. Contributions to the Study of SMS Spam Filtering: A New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng)*, 2011.
- UCI Machine Learning Repository, SMS Spam Collection. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Sanh, V., Debut, L., Chaumond, J., and Wolf, T. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC2 Workshop*, 2019.
- Ribeiro, M. T., Singh, S., and Guestrin, C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*, 2016.
- Lundberg, S. M. and Lee, S. A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 2017.
