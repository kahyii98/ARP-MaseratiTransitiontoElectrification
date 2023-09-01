# ARP - Maserati Transition to Electrification

## Project Overview
This applied research project aims to explore how younger audiences, specifically those in their teens and twenties, react on social media to Maserati's transition to electric vehicles.


---

## Repository Structure

The repository is divided into two main folders: `Pistonheads` and `Youtube`. Each folder contains specific Python notebooks related to data scraping, analysis, and modeling.

---

### 📁Pistonheads

#### `pistonheadsScraping.ipynb`
- **Purpose**: Scrapes data from the Pistonheads website about the Maserati Granturismo Folgore.
- **Output**: `pistonheads.xlsx` for further analysis.

---

### 📁YouTube

The Youtube folder is further divided into two sub-folders: `FormulaE` and `GranturismoFolgore`.

#### 📁FormulaE

##### `formulaEScraping.ipynb`
- **Purpose**: Scrapes YouTube comments related to Formula E.
- **Output**: FormulaE.xlsx for further analysis.

##### `formulaEAnalysis.ipynb`
- **Purpose**: Performs sentiment analysis and topic modelling on the well-structured scraped data `RobertaSentimentalFormulaEfinal.xlsx`.
- **Output**: Visualizations of comment volume and sentiment over time, and topic labels for Formula E June 2023.

---

#### 📁Granturismo Folgore

##### `preprocessingPosTag.ipynb`
- **Purpose**: Preprocesses the Granturismo Folgore data for POS tagging.
- **Output**: `AgeProcessedData.xlsx` ready for age prediction modeling.

##### `agePredictionTraining.ipynb`
- **Purpose**: Trains BERT and RoBERTa models for age prediction.
- **Output**: Trained models.

##### `naiveBayesBigrams.ipynb`
- **Purpose**: Implements a bigram model for age prediction.
- **Source**: [GitHub Repository](https://github.com/twistedTightly/NLP-Age-Classification/blob/master/src/naive_bayes_bigrams_class_return_probs.py)

##### `pos_tagger_class.ipynb` & `naiveBayesCombined.ipynb`
- **Purpose**: Additional code for`naiveBayesFinalModel.ipynb`.
- **Source**: [GitHub Repository](https://github.com/twistedTightly/NLP-Age-Classification)

##### `naiveBayesFinalModel.ipynb`
- **Purpose**: Serves as the final age prediction model using Naive Bayes. The model is provided by [ GitHub repository](https://github.com/twistedTightly/NLP-Age-Classification/blob/master/src/naive_bayes_combined.py).
- **Output**: Includes the overall distribution of sentiment analysis, age-predicted categories, and a stacked bar chart of sentiment within age categories. Note that the analysis (output) is original work and not part of the provided model.

##### `topicModelling.ipynb`
- **Purpose**: Performs topic modeling for Granturismo Folgore.
- **Output**: Visualizations and second-level topic modeling for each sentiment category.

---
