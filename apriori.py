import itertools

# Generate frequent itemsets from the given transactions
def apriori(transactions, support_threshold):
    # Count the occurrence of each item
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    # Filter out items that don't meet the support threshold
    frequent_items = {item: count for item, count in item_counts.items() if count >= support_threshold}
    
    # Keep track of the frequent itemsets
    frequent_itemsets = [set([item]) for item in frequent_items.keys()]
    
    # Continue to generate frequent itemsets until no new ones can be found
    while True:
        # Generate candidate itemsets by joining frequent itemsets
        candidate_itemsets = set()
        for itemset1 in frequent_itemsets:
            for itemset2 in frequent_itemsets:
                if itemset1 != itemset2:
                    candidate_itemset = itemset1.union(itemset2)
                    if len(candidate_itemset) == len(itemset1) + 1:
                        candidate_itemsets.add(candidate_itemset)
        
        # Count the occurrence of each candidate itemset
        itemset_counts = {}
        for transaction in transactions:
            for itemset in candidate_itemsets:
                if itemset.issubset(transaction):
                    itemset_counts[itemset] = itemset_counts.get(itemset, 0) + 1
        
        # Filter out itemsets that don't meet the support threshold
        frequent_itemsets = [itemset for itemset, count in itemset_counts.items() if count >= support_threshold]
        
        # Stop if no new frequent itemsets can be found
        if len(frequent_itemsets) == 0:
            break
    
    # Return the frequent itemsets and their counts
    frequent_itemset_counts = {tuple(itemset): itemset_counts[itemset] for itemset in frequent_itemsets}
    return frequent_itemset_counts

# Example usage
transactions = [
    {'bread', 'milk'},
    {'bread', 'diaper', 'beer', 'egg'},
    {'milk', 'diaper', 'beer', 'cola'},
    {'bread', 'milk', 'diaper', 'beer'},
    {'bread', 'milk', 'diaper', 'cola'}
]
support_threshold = 3

frequent_itemsets = apriori(transactions, support_threshold)
print(frequent_itemsets)
