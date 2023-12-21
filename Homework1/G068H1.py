from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from operator import add

# Reads the input file into an RDD of strings called rawData 
# (each 8-field row is read as a single string), and subdivides it into K partitions, 
# and prints the number of rows read from the input file (i.e., the number of elements of the RDD).
def printRDD(rdd):
    dataColl=rdd.collect()
    for row in dataColl:
        print(row)

# Given a list of ProductID,Popularity pairs, it prints its elements
def print_product_popularity(list_product_popularity):
    for elem in list_product_popularity:
        print(f"Product: {elem[0]} Popularity: {elem[1]}; ", end=" ")

# This method reads each line of the variable doc and for each string creates a tuple of (product_id,customer_id).
# To each tuple it assigns a key, which is generated concatenating the strings product_id + customer_id.
# A tuple is created only if the quantity is greater than zero and if the country is equal to the parameter S.
def to_key_prodCustomer(doc,S):
    pairs = {}
    for line in doc.split('\n'):
        transaction_id, product_id, desc, quantity, transaction_date,price, customer_id, country = line.split(',')
        if (int(quantity)>0 and (S==country or S=="all")):
            key = product_id + customer_id #string concatenation
            data = (product_id,int(customer_id))
            pairs[key]=data
    return [(key,pairs[key]) for key in pairs.keys()]

# Maps each partition's line to pair (product_id, customer_id), having product_id as key and setting the value to 
# the occurrences of customers for that product_id
def to_prodCustomer_with_partitions(partitions):
    pairs = {}
    for index,line in enumerate(partitions):
        product_id, customer_id = line[0],line[1]
        key = product_id 
        if key not in pairs.keys():
            pairs[key] = 1
        else:
            pairs[key] += 1

    return [(key,pairs[key]) for key in pairs.keys()]

def main():
    
    #SPARK SETUP
    # We create a SparkConf() object.
    conf = SparkConf().setAppName('Homework1').setMaster("local[*]")

    #Then we give the conf object to an object SparkContext
    sc = SparkContext(conf=conf)
    
    # INPUT READING

    # Number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Number of product with highest popularity to find
    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"

    # Country
    S = sys.argv[3]

    # Path
    data_path = sys.argv[4]

    # First point
    rawData = sc.textFile(data_path,minPartitions=K).cache()
    rawData.repartition(numPartitions=K)
    num_rows = rawData.count()
    print(f"Number of rows = {num_rows}")

    # Second point
    productCustomer = (rawData.flatMap(lambda x: to_key_prodCustomer(x, S)) # MAP PHASE -> (key, (p,c))
                       .reduceByKey(lambda x,y: y) # REDUCE PHASE: removes duplicates and returns distinct pairs (p,c)
                       .values()
                    )

    print(f"Product-Customer Pairs: {productCustomer.count()}")
    
    #Third point
    productPopularity1 = (productCustomer.mapPartitions(lambda x: to_prodCustomer_with_partitions(x)) # MAP PHASE -> (p,occurrences of clients in partition)
                        .groupByKey().mapValues(sum)  # REDUCE PHASE: groups by product key (p, occurrences of clients) -> (p,sum(occurrences))
    )
    
    #Fourth point    
    productPopularity2 = (productCustomer.map(lambda x: (x[0],1)) # MAP PHASE: filters by country and quantity -> (p,1)
                        .reduceByKey(add)  #REDUCE PHASE: groups by product key (p,1+1+..1) -> (p,sum)
    )
        
    #Fifth point
    if int(H) > 0:
        top_H = (productPopularity1.map(lambda x : (x[1],x[0])) # (prodID, popularity) -> (popularity,prodID)
                .sortByKey(False, numPartitions=K) # Groups partitions by popularity and sorts them in descending order
                .map(lambda x: (x[1],x[0])) # (popularity,prodID)->(prodID,popularity)
                .take(int(H)) # takes top H
        )
        print(f"Top {len(top_H)} Products and their Popularities    ")
        print_product_popularity(top_H)
    else:
        #Sixth point
        print("productPopularity1: ")
        sorted_product_popularity1 = productPopularity1.sortByKey(True,numPartitions=K).collect()
        print_product_popularity(sorted_product_popularity1)
        print("\nproductPopularity2: ")
        sorted_product_popularity2 = productPopularity2.sortByKey(True,numPartitions=K).collect()
        print_product_popularity(sorted_product_popularity2)

if __name__ == "__main__":
	main()