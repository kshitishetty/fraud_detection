# Credit Card Fraud Detection

A small weekend-style project that uses machine learning to spot fraudulent credit card transactions. You enter some transaction details, it tells you if something smells off.

---

## Getting it running

You'll need Python 3.8 or above. Install the dependencies first:

```bash
pip install flask flask-cors pandas scikit-learn
```

Then just start the server:

```bash
python app.py
```

It'll take a few seconds on startup — it's training the model right then and there, not loading a pre-saved one. Once Flask says it's running, open `index.html` in your browser and you're all set.

---

## Using it

There's a simple form where you enter the transaction amount and a few values (`V1`, `V2`, `V3`). Hit the button and you'll get either a green light or a fraud warning, along with how confident the model is in that call.

That's genuinely it. No accounts, no setup beyond what's above.

---

## Under the hood 

The model trains on a dataset of ~285,000 real credit card transactions from Kaggle. Most of the features (`V1` through `V28`) are anonymized using PCA — the original values are hidden for privacy reasons — but `Amount` and `Time` are left raw.

On startup, `app.py` loads all that data, scales the features, and trains a Random Forest with 100 trees. From there, the `/predict` endpoint takes in transaction data and returns a verdict.

The frontend only collects `Amount`, `V1`, `V2`, and `V3` from you — everything else quietly defaults to `0`. It's enough to play around with, but it's not a full picture.

---

## Hitting the API directly

If you want to skip the UI:

```
POST http://127.0.0.1:5000/predict
```

```json
{
  "Amount": 149.62,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.54
}
```

Any feature you don't include just defaults to `0`. You'll get back something like:

```json
{
  "fraud": 1,
  "probability": 94.3
}
```

`fraud: 1` means it thinks something's wrong. `probability` is how sure it is, as a percentage.

---

## 

The dataset is pretty lopsided — only about 0.17% of transactions are actually fraud. So the model has seen way more legitimate transactions than fraudulent ones, which is realistic but means it can be overconfident. A 99% confidence score doesn't mean it's infallible.

The model also retrains every time you restart the server. It's not slow, but there's no saved file — it starts fresh each time.
