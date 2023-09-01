# ARP - Maserati Transition to Electrification

## Project Overview
This applied research project aims to explore how younger audiences, specifically those in their teens and twenties, react on social media to Maserati's transition to electric vehicles.

---

## Repository Structure

The repository is divided into two main folders: `Pistonheads` and `Youtube`. Each folder contains specific Python notebooks related to data scraping, analysis, and modeling.

---

### Pistonheads

#### `pistonheadsScraping.ipynb`
- **Purpose**: Scrapes data from the Pistonheads website about the Maserati Granturismo Folgore.
- **Output**: Raw data for further analysis.

---

### Youtube

The Youtube folder is further divided into two sub-folders: `FormulaE` and `GranturismoFolgore`.

#### FormulaE

##### `formulaEScraping.ipynb`
- **Purpose**: Scrapes Youtube comments related to Formula E.
- **Output**: Raw data for further analysis.

##### `formulaEAnalysis.ipynb`
- **Purpose**: Performs sentiment analysis and topic modeling on the scraped data.
- **Output**: Visualizations of comment volume and sentiment over time, and topic labels for June 2023.

---

#### Granturismo Folgore

##### `preprocessingPosTag.ipynb`
- **Purpose**: Preprocesses the Granturismo Folgore data for POS tagging.
- **Output**: Data ready for age prediction modeling.

##### `agePredictionTraining.ipynb`
- **Purpose**: Trains BERT and RoBERTa models for age prediction.
- **Output**: Trained models.

##### `naiveBayesBigrams.ipynb`
- **Purpose**: Implements a bigram model for age prediction.
- **Source**: [GitHub Repository](https://github.com/twistedTightly/NLP-Age-Classification/blob/master/src/naive_bayes_bigrams_class_return_probs.py)

##### `pos_tagger_class.ipynb` & `naiveBayesCombined.ipynb`
- **Purpose**: Additional code for Naive Bayes model.
- **Source**: [GitHub Repository](https://github.com/twistedTightly/NLP-Age-Classification)

##### `naiveBayesFinalModel.ipynb`
- **Purpose**: Final age prediction model using Naive Bayes.
- **Output**: Overall distribution of sentiment analysis, age predicted categories, and a stacked bar chart of sentiment within age categories.

##### `topicModelling.ipynb`
- **Purpose**: Performs topic modeling for Granturismo Folgore.
- **Output**: Visualizations and second-level topic modeling for each sentiment category.

---
