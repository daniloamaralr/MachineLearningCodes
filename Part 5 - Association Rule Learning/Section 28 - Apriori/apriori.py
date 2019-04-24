#Apriori
 
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the mall dataset with pandas
dataset = pd.read_csv('../../datasets/Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j])for j in range(0,20)])
    
# Training a Apriori on the Dataset
# Determine the min_support = 3*7 / 7500
from apyori import apriori
rules = apriori(transactions,min_support  = 0.003,min_confidence = 0.2,min_lift= 3 , min_length = 2)

# Visualising the results
results = list(rules)