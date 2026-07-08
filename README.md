# 💳 Credit Card Fraud Detection (KNN) — Vercel Deployment

A machine-learning web app that classifies credit-card transactions as
**fraudulent** or **valid** using a **K-Nearest-Neighbors** model trained on the
[ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 real transactions, 492 frauds). The trained model is served through a
lightweight **Python serverless function** on **Vercel** with a static frontend.

## ✨ Highlights

- **~88% balanced accuracy** (plus precision, recall, F1 and ROC-AUC) on a held-out test set.
- Handles extreme class imbalance (0.172% fraud) via **random under-sampling**.
- **Zero-scikit-learn runtime**: the serverless API re-implements KNN inference in
  pure NumPy, so it stays well under Vercel's size limit and cold-starts fast.
- Clean, responsive UI with real sample transactions from the dataset.

## 🗂️ Project structure

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

## 🚀 Quick start (local)

```powershell
# 1. Create the dataset folder and download the data (~144 MB)
mkdir data
curl -L -o data/creditcard.csv "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# 2. Install training dependencies
pip install scikit-learn pandas numpy

# 3. Train the model (writes api/model.npz and model/metadata.json)
python model/train.py
```

## ☁️ Deploy to Vercel

1. Push this repo to GitHub (the `data/` folder is git-ignored — only
   `api/model.npz` is needed at runtime, and it is committed).
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import the repo.
3. Framework preset: **Other**. No build command needed. Deploy.

Or with the CLI:

```powershell
npm i -g vercel
vercel        # preview
vercel --prod # production
```

The frontend is served from `public/index.html`; the API lives at `/api/predict`.

## 🔌 API

`POST /api/predict`

```json
{ "features": [V1, V2, "...", V28, Amount] }
```

Response:

```json
{ "prediction": 1, "label": "fraud", "fraud_probability": 0.86, "k": 5, "fraud_votes": 4 }
```

## 📊 Dataset & method

- **Source:** Machine Learning Group – ULB, via Kaggle (`mlg-ulb/creditcardfraud`).
- Features `V1…V28` are PCA components; `Amount` is the transaction value.
- Because raw accuracy is misleading on such an imbalanced set, the data is
  **balanced by under-sampling** the majority class so accuracy is meaningful,
  then a `StandardScaler` + `KNeighborsClassifier` pipeline is trained with a
  small grid search over `k`.

> ⚠️ Educational demo only — not intended for real financial decision-making.
