# Credit Card Fraud Detection (KNN) — Vercel Deployment

A machine-learning web app that classifies credit-card transactions as
**fraudulent** or **valid** using a **K-Nearest-Neighbors** model trained on the
[ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 real transactions, 492 frauds). The trained model is served through a
lightweight **Python serverless function** on **Vercel** with a static frontend.

## Highlights

- **~88% balanced accuracy** (plus precision, recall, F1 and ROC-AUC) on a held-out test set.
- Handles extreme class imbalance (0.172% fraud) via **random under-sampling**.
- **Zero-scikit-learn runtime**: the serverless API re-implements KNN inference in
  pure NumPy, so it stays well under Vercel's size limit and cold-starts fast.
- Clean, responsive UI with real sample transactions from the dataset.

##  Results

Evaluated on a held-out, balanced test set (246 transactions, 50/50 split):

| Metric | Score |
| --- | --- |
| Accuracy | **91.9%** |
| Precision | 95.6% |
| Recall | 87.8% |
| F1-score | 91.5% |
| ROC-AUC | 0.944 |

Confusion matrix:

|  | Predicted valid | Predicted fraud |
| --- | --- | --- |
| **Actual valid** | 118 (TN) | 5 (FP) |
| **Actual fraud** | 15 (FN) | 108 (TP) |

## Project structure

```
fraud/
├── api/
│   ├── predict.py        # Vercel serverless function (NumPy KNN inference)
│   └── model.npz         # exported model artifact (created by training)
├── model/
│   ├── train.py          # training + evaluation pipeline (scikit-learn)
│   └── metadata.json     # metrics, k, feature list (created by training)
├── public/
│   └── index.html        # frontend
├── data/                 # dataset lives here (git-ignored, downloaded)
├── requirements.txt      # runtime deps for the serverless function (numpy)
├── vercel.json
└── README.md
```


## Dataset & method

- **Source:** Machine Learning Group – ULB, via Kaggle (`mlg-ulb/creditcardfraud`).
- Features `V1…V28` are PCA components; `Amount` is the transaction value.
- Because raw accuracy is misleading on such an imbalanced set, the data is
  **balanced by under-sampling** the majority class so accuracy is meaningful,
  then a `StandardScaler` + `KNeighborsClassifier` pipeline is trained with a
  small grid search over `k`.

