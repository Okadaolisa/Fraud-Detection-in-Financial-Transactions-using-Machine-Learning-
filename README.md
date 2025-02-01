# Project Title: **Dissertation Analysis Notebook**

## Overview
This repository contains a Jupyter Notebook designed for in-depth analysis and machine learning experimentation as part of a dissertation project. The notebook incorporates text processing, statistical analysis, and predictive modeling to explore and derive insights from data.

---

## Features
- **Text Processing**: Tokenization, stopword removal, and feature extraction from textual data.
- **Data Preprocessing**: Label encoding, normalization, and transformation for modeling.
- **Exploratory Data Analysis (EDA)**: Visualizations using Seaborn and Plotly.
- **Machine Learning Models**: Implementation of logistic regression, Naïve Bayes, and ensemble models.

---

## Prerequisites
Ensure you have the following installed before running the notebook:
- Python 3.10 or later
- Jupyter Notebook or Jupyter Lab
- Required Python libraries:
  ```bash
  pip install pandas numpy seaborn plotly nltk scikit-learn scipy
  ```

---

## Key Libraries Used
```python
import re
import string
import scipy
import pickle
import os, glob
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
```

---

## Dataset
- **File Name**: The dataset is placed in the same directory as the notebook.
- **Contents**: Contains textual and numerical data for analysis and modeling.

---

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/dissertation-analysis.git
   ```
2. Navigate to the project folder:
   ```bash
   cd dissertation-analysis
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "260764 Dissertation.ipynb"
   ```
4. Run the notebook cells sequentially to execute the analysis.

---

## Example Workflow
1. Import necessary libraries.
2. Load and preprocess the dataset:
   ```python
   df = pd.read_csv("your_dataset.csv")
   ```
3. Perform exploratory data analysis:
   - Generate summary statistics
   - Visualize data distributions
4. Train and evaluate machine learning models:
   - Logistic Regression
   - Naïve Bayes
   - Ensemble Methods
5. Interpret the results and derive insights.

---

## Outputs
- **Visualizations**: Heatmaps, bar charts, interactive plots.
- **Statistical Insights**: Data correlations, hypothesis testing.
- **Predictive Models**: Trained classifiers and performance metrics.

---

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

---

## Acknowledgments
Special thanks to the open-source community and researchers for their contributions to machine learning and data science.

---

## Contact
For any questions or support, please contact:
- **Name**: Adaolisa Okafor
- **Email**: adaolisa.margaret@gmail.com
- **GitHub**: [Okadaolisa](https://github.com/Okadaolisa)


