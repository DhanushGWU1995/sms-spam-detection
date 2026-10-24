# Project Proposal: SMS Spam Detection

## Team

Add our names and who is doing what before we turn this in.

## Problem

We want to tell spam SMS from normal ones (ham). Keyword rules are easy to dodge, and they also block normal texts that happen to say "free" or "call". The plan is to train a classifier on the message text and then check which words the model is using.

## Dataset

The [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) has 5,574 English messages: 4,827 ham and 747 spam. The texts are short and full of abbreviations.

We will drop duplicate messages before splitting, so the same text cannot land in both train and test. The split is 80/10/10, stratified, with one seed for every model.

Spam is only 13.4% of the data. Always predicting ham is already about 87% accurate, so we will compare models with spam precision, recall, F1, and PR-AUC.

## Approach

Same task, three models, then explanations.

| Model | What we will train | Why it is in the project |
| --- | --- | --- |
| TF-IDF + logistic regression | Word and character n-grams, class-weighted logistic regression | Classical baseline |
| GRU | Embedding plus one GRU in PyTorch, class-weighted loss | RNN |
| DistilBERT | Fine-tune `distilbert-base-uncased`, max length 128, a few epochs | Transformer |
| LIME and SHAP | Coefficients, LIME on single messages, SHAP on the linear model | Interpretability |

Cleaning stays light. Digits, currency, and short codes are spam clues, so the linear model and the GRU keep them. DistilBERT reads the raw text through its own tokenizer.

Each model gets its own spam threshold, chosen on the validation set by spam F1. We only look at the test labels for the final table.

## Demo

`Code/app_deploy.py` is a Streamlit app. After the notebook saves the models, we can paste one SMS and see the three predictions side by side, with the spam probability for each.

## What we will turn in

| Folder | What goes here |
| --- | --- |
| `Code/` | Notebook, demo, data, saved models |
| `Group-Proposal/` | This proposal |
| `Final-Group-Presentation/` | Slides |
| `Final-Group-Project-Report/` | Write-up with the table, plots, and a few error examples |

## Evaluation

For each model:

- accuracy
- spam precision, recall, and F1
- PR-AUC
- confusion matrix

Then a short look at missed spam and blocked ham, using the same example texts for LIME and SHAP.

## References

- Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. Contributions to the Study of SMS Spam Filtering: A New Collection and Results. ACM DocEng, 2011.
- UCI Machine Learning Repository, SMS Spam Collection. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Sanh, V., Debut, L., Chaumond, J., and Wolf, T. DistilBERT, a distilled version of BERT. NeurIPS EMC2 Workshop, 2019.
- Ribeiro, M. T., Singh, S., and Guestrin, C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD, 2016.
- Lundberg, S. M. and Lee, S. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.
