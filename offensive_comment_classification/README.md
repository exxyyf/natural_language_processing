I used logistic regression + vectorizers, then I performed transfer learning with BERT in order to classify offensive and neutral comments. Soon I'll expand this to multilabel classification.
<br></br>
[**Jupyter Notebook**](https://github.com/exxyyf/natural_language_processing/blob/main/offensive_comment_classification/toxic_comments_mail.ipynb)
<br></br>
**Stack**: `Scikit-learn`, `nltk`, `transformers`, `torch`, `pandas`, `numpy`, `seaborn`

Test ROC-AUC: **0.9938746929905539**
