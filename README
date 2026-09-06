# 📱 SMS Spam Detection — Machine Learning Mini-Project

A complete, production-ready Natural Language Processing (NLP) pipeline built to automatically classify SMS messages into **Ham** (normal) or **Spam** (unwanted/promotional). This project processes raw text strings, converts them into statistical numerical features, and trains a Logistic Regression engine to accurately predict hidden threats.

---

## 🎯 Problem Statement

Given a dataset containing thousands of real-world text messages, the objective is to build a high-performance classification engine capable of distinguishing safe communications from spam attempts.

### 📌 Dataset Specifications

The input dataset (`spam.csv`) contains two baseline data vectors:

- **Category:** Explicit categorical labels (`ham` or `spam`).
- **Message:** The raw text content of the SMS string.

---

## 🧩 Step-by-Step Task Execution

### Task 1 — Load and Explore the Dataset

- Imported fundamental core data science modules: `pandas`, `matplotlib`, `seaborn`, and `sklearn`.
- Parsed the source `spam.csv` storage file directly into a memory-mapped Data Frame using `pd.read_csv()`.
- Checked structural vectors using `df.shape` and investigated head samples using `df.head()`.

### Task 2 — Visualize the Dataset 📊

- Constructed a clean statistical `sns.countplot` categorical distribution bar chart.
- **Analysis Question Resolution:** The analysis shows that `ham` (safe messages) represents the vast majority of the real-world dataset samples (~86%), pointing to a heavily imbalanced dataset distribution.

### Task 3 — Prepare the Data

- Evaluated labels and mapped the strings directly to structural binaries: **Ham → 0**, **Spam → 1**.
- Explicitly separated the memory matrices into input features (`X` representing the message texts) and targets (`y` representing encoded binary classifications).

### Task 4 — Split the Dataset

- Segmented the sample populations using an **80/20 train-test configuration rule**.
- **Analysis Question Resolution:** We split the data vectors to securely establish an independent test set. This evaluates the generalization performance of the model on unseen data inputs, proving whether it has actually learned underlying word probability vectors rather than simply memorizing the patterns.

### Task 5 — Convert Text into Numbers 🔢

- Initialized a token frequency parsing pipeline via `CountVectorizer`.
- Explicitly cleaned vocabulary metrics by eliminating generic `english` stopwords.
- **Conceptual Architecture Insight:** Machine Learning models are purely mathematical matrix calculators. They cannot interpret variable-length alphabetic strings directly. Converting text documents into numerical bag-of-words arrays gives the model numeric vectors it can multiply against internal structural weights.

### Task 6 & 7 — Train a Machine Learning Model & Make Predictions 🤖

- Formulated an optimized optimization plane using a `LogisticRegression` classification engine.
- Transmitted the numerical vocabulary matrix into the model using `.fit()` to lock in feature weights, and stored the test set classifications within a structured `predictions` variable.

### Task 8 — Evaluate the Model 📈

- Calculated the structural prediction accuracy score using `accuracy_score()`.
- Rendered a clean `confusion_matrix` heatmap displaying precise counts for **True Negative (0,0)**, **False Positive (0,1)**, **False Negative (1,0)**, and **True Positive (1,1)** errors.
- **Analysis Question Resolution:** The dynamic addition of correct matrix outcomes (Diagonal sum of True Hams + True Spams) explicitly benchmarks the precise counts of correctly caught test elements.

### Task 9 — Test Your Own SMS 📱

- Configured an interactive execution terminal utilizing `input()`.
- Applied matching structural changes onto raw test input text frames via `.transform()` to generate live, adaptive prediction outputs: `📩 HAM MESSAGE` or `🚨 SPAM MESSAGE`.

---

## 💻 Tech Stack & Dependencies

- **Programming Environment:** Python (`.ipynb` Jupyter Notebook Engine)
- **Core Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📁 Repository Structure

```text
NLP Project/
├── spam.csv              # Source SMS Dataset
├── main.ipynb            # Interactive Cell-by-Cell Pipeline Notebook
└── README.md             # Project Performance & Structural Log
```

---

_Developed as a core exploration blueprint into text mining, feature extraction, and linear model classification paradigms._
