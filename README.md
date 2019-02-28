# HU Kaggle Competition – Clothing Returns Prediction

The task was predicting whether customers will return their online clothing order. (Binary classification) We were given 150k rows of labelled training data with 14 columns.
For my submission, I used machine learning algorithms such as Random Forests, XGBoost, Early-Stopping Neural Networks, and employed a selective heterogenous Ensemble to further improve my score.

For the Business Analytics and Data Science course at the chair of Information Systems, Humboldt-Universität zu Berlin, there was an [in-class data science competition on Kaggle](https://www.kaggle.com/c/bads1718). The target was to predict customers who would return their order (binary classification). 150k rows of labelled data were available as training data. For this task, machine learning algorithms such as Random Forests, XGBoost, Early-Stopping Neural Networks, were utilised, along with selective heterogenous ensembling for final score improvements. Scoring for the competition was based on AUC predictive accuracy. After the competition and for the final deliverable of the project, a price sensitive cost function is employed to look at the real-world business tradeoff between:
- Falsely determining a customer to be a non-returner and selling to someone who returns their item / order.

- Being overly conservative and not making sales to incorrectly classified customers who would have otherwise kept their items and possibly been a return customer.

Final code is in the 'final_code directory' and is separated between the data cleaning script and the modeling script. All code was written in Python within the Atom / Hydrogen Jupyter Notebook environment.

My efforts earned me position 4 of 118 in the class.

<img src = "https://github.com/alextruesdale/clothing-returns-prediction/blob/master/repository_media/kaggle.png" alt = "Kaggle Competition" title = "Kaggle Competition" align = "center" width = "830" />
