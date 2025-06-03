
# ✈️ British Airways Reviews Sentiment Analysis

This project focuses on performing exploratory data analysis (EDA) and sentiment analysis on customer reviews of **British Airways** using Natural Language Processing (NLP) techniques in Python.

---

## 📦 Project Setup

### 1. Install Required Libraries

```bash
pip install pandas matplotlib seaborn wordcloud nltk textblob
```

### 2. Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
```

---

## 📥 Data Loading

* The dataset (`british-airways-review.csv`) was uploaded and loaded into a pandas DataFrame.
* Columns include: `Reviewer`, `Date`, `Rating`, `Review`, `heading1`.

---

## 🧹 Data Cleaning & Preprocessing

### ✅ Actions Taken:

* Checked for missing data using `df.info()`.
* Converted text to lowercase.
* Removed domain-specific stopwords (`flight`, `airline`, `ba`, etc.).
* Tokenized text and filtered out punctuation and stopwords.

```python
custom_stopwords = set(stopwords.words('english'))
custom_stopwords.update(['ba', 'flight', 'airline', 'flights', 'airways', 'trip', 'verified', '|','british','us','would','one','cabin'])
```

---

## ☁️ Word Cloud

A word cloud was generated to visualize the most frequent words in customer reviews.

![Word Cloud Example](#)
![alt text](image.png)
---

## 📊 Most Common Words

A bar chart showing the top 10 most frequent words:

| Word    | Frequency |
| ------- | --------- |
| service | High      |
| staff   | High      |
| seat    | Moderate  |
| food    | Moderate  |
| delay   | Moderate  |

---

## ✍️ Review Length Analysis

* A histogram showed the distribution of review lengths.
* Most reviews contained **50–150 words**, suggesting moderate review depth.

---

## 😊 Sentiment Analysis

### Sentiment Metrics:

* **Polarity** (range: -1 to 1): how positive/negative a review is.
* **Subjectivity** (range: 0 to 1): how subjective/personal a review is.

```python
df['polarity'] = df['cleaned_reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['cleaned_reviews'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
```

### 🔍 Insight:

Most reviews cluster around **neutral to mildly positive polarity**, indicating general satisfaction but also variability in experience.

---

## 📉 Sentiment Distribution

### TextBlob and VADER Analysis:

Both VADER and TextBlob tools were used to classify sentiments of both `heading1` and `Review`.

```python
# Classification:
compound >= 0.05 → Positive  
compound <= -0.05 → Negative  
otherwise → Neutral
```

### Distribution Pie Charts:

* **VADER Review Sentiment**:

  * Positive: \~50%
  * Neutral: \~30%
  * Negative: \~20%

* **TextBlob Review Sentiment**:

  * Similar distribution, with slightly fewer negative reviews.

---

## 📈 Scatter Plot: Polarity vs. Subjectivity

* Displayed how opinionated reviews (subjectivity) relate to their sentiment polarity.
* No strong correlation observed.
* Most reviews are moderately subjective and clustered near 0 polarity.

---

## 🔍 Final Insights

* **Common Themes**: Service, food, seating comfort, delays.
* **Sentiment Summary**:

  * Majority reviews are **positive or neutral**, showing general customer satisfaction.
  * A significant minority express negative sentiments, mainly concerning service delays or staff behavior.
* **Dual Sentiment Tools**: Using both VADER and TextBlob provided validation and robustness in sentiment classification.

---

## 📁 Project Structure

```
airline-review-analysis/
├── british-airways-review.csv
├── sentiment_analysis_notebook.ipynb
├── README.md
```

---

## ✅ Future Improvements

* Perform **topic modeling** (LDA) to identify underlying themes.
* Integrate **interactive visualizations** with Plotly or Dash.
* Use a **machine learning model** to classify reviews as positive/negative/neutral.

---

## 📚 References

* [NLTK](https://www.nltk.org/)
* [TextBlob](https://textblob.readthedocs.io/en/dev/)
* [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
* [WordCloud](https://github.com/amueller/word_cloud)

