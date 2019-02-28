# HU Kaggle Competition – Clothing Returns Prediction

For the Business Analytics and Data Science course at the chair of Information Systems, Humboldt-Universität zu Berlin, all student participated in an [in-class data science competition on Kaggle](https://www.kaggle.com/c/bads1718). The target was to predict customers who would return their order (binary classification) with 150k rows of labelled data given as training data. For this task, machine learning algorithms such as Random Forest, XGBoost, Early-Stopping Neural Networks, were utilised, along with selective heterogenous ensembling for final score improvements. In addition to achieving a high AUC score (Kaggle scoring criteria), there was the additional task of using a profit sensitive model to take into account asymmetric error costs and thus move from predictive to prescriptive modelling.

The real-world tradeoff handled by this modeling is then between:
- Falsely determining a customer to be a non-returner, selling to someone who does indeed return their item / order.

- Being overly conservative and not making sales to incorrectly classified customers who would have otherwise kept their items (and possibly been a return customer).

Final code is in the 'final_code directory' and is separated between the data cleaning script and the modeling script. All code was written in Python within the Atom / Hydrogen Jupyter Notebook environment.

My efforts earned me position 4 of 118 in the class.

<img src = "https://github.com/alextruesdale/clothing-returns-prediction/blob/master/repository_media/kaggle.png" alt = "Kaggle Competition" title = "Kaggle Competition" align = "center" width = "830" />
