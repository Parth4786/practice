#!pip install apyori 
#Apriori Algorithm 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from apyori import apriori 
min_supp = float(input("Enter Minimum Support: ")) 
min_conf = float(input("Enter Minimum Confidence: ")) 
store_data = pd.read_csv('store_data.csv')
store_data = pd.read_csv('store_data.csv', header=None) 
store_data.head() 
records = [] 
for i in range(0, 7501): 
    records.append([str(store_data.values[i,j]) for j in range(0, 20)]) 
association_rules = apriori(records, min_support=min_supp, min_confidence=min_conf, min_lift=3, min_length=2) 
association_results = list(association_rules) 
for item in association_results: 
    # first index of the inner list 
    # # Contains base item and add item 
    pair = item[0] 
    items = [x for x in pair] 
    print("Rule: " + items[0] + " -> " + items[1])
    #second index of the inner list 
    print("Support: " + str(item[1])) 
    #third index of the list located at 0th 
    # #of the third index of the inner list 
    print("Confidence: " + str(item[2][0][2])) 
    print("Lift: " + str(item[2][0][3])) 
    print("=====================================")
    
#FP Tree 
# !pip install pandas 
# !pip install numpy 
# !pip install plotly 
# !pip install mlxtend 
# !pip install mlxtend.frequent_patterns 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import fpgrowth 
from mlxtend.frequent_patterns import association_rules 
# dataset 
dataset = pd.read_csv("store_data.csv") 
# printing the shape of the dataset 
dataset.shape 
dataset.head() 
# Gather All Items of Each Transactions into Numpy Array 
transaction = [] 
for i in range(0, dataset.shape[0]): 
    for j in range(0, dataset.shape[1]): 
        transaction.append(dataset.values[i,j]) 
# converting to numpy array 
transaction = np.array(transaction) 
print(transaction)
# Transform Them a Pandas DataFrame 
df = pd.DataFrame(transaction, columns=["items"]) 
# Put 1 to Each Item For Making Countable Table, to be able to perform Group By 
df["incident_count"] = 1 
# Delete NaN Items from Dataset 
indexNames = df[df['items'] == "nan" ].index 
df.drop(indexNames , inplace=True) 
# Making a New Appropriate Pandas DataFrame for Visualizations 
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index() 
# # to have a same origin 
df_table["all"] = "Top 50 items" 
# creating tree map using plotly 
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count', color=df_table["incident_count"].head(50), hover_data=['items'], color_continuous_scale='Blues', ) 
# ploting the treemap 
fig.show() 
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array 
transaction = [] 
for i in range(dataset.shape[0]): 
    transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])

# creating the numpy array of the transactions 
transaction = np.array(transaction) 
# initializing the transactionEncoder 
te = TransactionEncoder() 
te_ary = te.fit(transaction).transform(transaction) 
dataset = pd.DataFrame(te_ary, columns=te.columns_) 
# dataset after encoded 
dataset.head() 
#running the fpgrowth algorithm 
res=fpgrowth(dataset,min_support=0.05, use_colnames=True) 
# printing top 10 
res.head(10) 
# creating asssociation rules 
res=association_rules(res, metric="lift", min_threshold=1) 
# printing association rules 
print(res) 
# Sort values based on confidence 
res.sort_values("confidence",ascending=False)