from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import io
df=pd.read_csv('groceries.csv')
transactions=[]
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,len(df.columns))])
te=TransactionEncoder()

te_array=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_array, columns=te.columns_)
print(df)

#df.dropna(subset=['colname'])
#df.dropna()

freq_items = apriori(df, min_support=0.5, use_colnames=True)



print(freq_items)

rules=association_rules(freq_items,metric='support',min_threshold=0.05)

rules=rules.sort_values(['support','confidence'],ascending=[False,False])

print(rules)