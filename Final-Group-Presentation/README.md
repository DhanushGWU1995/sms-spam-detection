# Final presentation

Put the slides in this folder (PowerPoint or PDF).

Here's the order we're planning:

1. The task: spam vs ham, and why it's still a problem.
2. The data: how we combined UCI and Super SMS, the class balance (about 61% ham / 39% spam), and why we removed duplicates and conflicting labels before splitting.
3. The three models: TF-IDF + logistic regression, GRU, and DistilBERT.
4. Results: the comparison table and confusion matrices from `Code/main.ipynb`. DistilBERT did best (F1 0.992).
5. LIME and SHAP: which words each model relied on, including the masked numbers (`xxx`) and placeholders (`num`) that come from how each dataset was anonymized.
6. Mistakes: a few examples of spam we missed and ham we wrongly blocked.
7. Live demo of `Code/app_deploy.py`.
