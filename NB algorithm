import pandas as pd
import numpy as np
import random
import math
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

# Load in the training set .csv
training_set = pd.read_csv("trg.csv")
training_set.head()

# Process the text, find a 'good model' with cross-validation
print("Text processing...")
ts = training_set["abstract"].str.split(" ")

i = 0
new_ts = []
while (i < ts.size):
    new_tt = [word for word in ts[i] if word not in en_stops]
    new_ts.append(np.unique(new_tt))
    i+= 1


# Train the NBC with this data
print("Training the NBC...")
cl = training_set["class"]
B = 0
freqB = pd.DataFrame({' ':[0]})
A = 0
freqA = pd.DataFrame({' ':[0]})
V = 0
freqV = pd.DataFrame({' ':[0]})
E = 0
freqE = pd.DataFrame({' ':[0]})
i = 0
while (i < cl.size):
    if cl[i] == "B":
        B+= 1
    elif cl[i] == "A":
        A+= 1
    elif cl[i] == "V":
        V+= 1
    elif cl[i] == "E":
        E+= 1
    i+= 1
pB = B/cl.size
pA = A/cl.size
pV = V/cl.size
pE = E/cl.size
"""
# put words into class dataframe. update frequency of word
# comment out once sent to csv files to save time
i = 0
while (i < cl.size):
    for word in new_ts[i]:
        if cl[i] == "B":
            if word in freqB:
                freqB.at[0, word] = 1 + freqB.iloc[0][word]
            else:
                freqB[word] = 1   
        if cl[i] == "A":
            if word in freqA:
                freqA.at[0, word] = 1 + freqA.iloc[0][word]
            else:
                freqA[word] = 1

        if cl[i] == "E":
            if word in freqE:
                freqE.at[0, word] = 1 + freqE.iloc[0][word]
            else:
                freqE[word] = 1
                            
        if cl[i] == "V":
            if word in freqV:
                freqV.at[0, word] = 1 + freqV.iloc[0][word]
            else:
                freqV[word] = 1
    i+=1
    print(i)

freqA.to_csv("tst_A.csv", index=False)
freqE.to_csv("tst_E.csv", index=False)
freqV.to_csv("tst_V.csv", index=False)
freqB.to_csv("tst_B.csv", index=False)
"""
#delete infrequent words
fA = pd.read_csv("tst_A.csv")
fA.head()
fB = pd.read_csv("tst_B.csv")
fB.head()
fE = pd.read_csv("tst_E.csv")
fE.head()
fV = pd.read_csv("tst_V.csv")
fV.head()

num_arr = [1,2,3,4,5,6,7,8,9]

for word in fV:
    if fV.loc[0, word] in num_arr:
        fV.drop(word, axis=1, inplace=True)

for word in fA:
    if fA.loc[0, word] in num_arr:
        fA.drop(word, axis=1, inplace=True)

for word in fB:
    if fB.loc[0, word] in num_arr:
        fB.drop(word, axis=1, inplace=True)

for word in fE:
    if fE.loc[0, word] in num_arr:
        fE.drop(word, axis=1, inplace=True)

fA.to_csv("freq_A.csv", index=False)
fE.to_csv("freq_E.csv", index=False)
fV.to_csv("freq_V.csv", index=False)
fB.to_csv("freq_B.csv", index=False)


fA = pd.read_csv("freq_A.csv")
fA.head()
fB = pd.read_csv("freq_B.csv")
fB.head()
fE = pd.read_csv("freq_E.csv")
fE.head()
fV = pd.read_csv("freq_V.csv")
fV.head()

fA.drop(' ',axis=1,inplace=True)
fB.drop(' ',axis=1,inplace=True)
fV.drop(' ',axis=1,inplace=True)
fE.drop(' ',axis=1,inplace=True)

# total unique words
i = 0
size_ts = []
while (i < cl.size):
    for word in new_ts[i]:
        if(word not in size_ts) and (word in fA or word in fB or word in fV or word in fE):
            size_ts.append(word)
    i+=1

# denominators
denomA = fA.values.sum() + len(size_ts)
denomB = fB.values.sum() + len(size_ts)
denomV = fV.values.sum() + len(size_ts)
denomE = fE.values.sum() + len(size_ts)

i = 0
while (i < cl.size):
    for word in new_ts[i]:
            if word in fA:
                if fA.at[0,word] > 1:
                    condA = (float(fA.at[0,word] + 1) / float(denomA))
                    fA[word] = condA
            elif word in fE or word in fV or word in fB:
                condA = (0 + 1 / denomA)
                fA[word] = condA
    i+=1

i=0 
while (i < cl.size):
    for word in new_ts[i]:
        if word in fB:
                if fB.at[0,word] > 1:
                    condB = (float(fB.at[0,word] + 1) / float(denomB))
                    fB[word] = condB
        elif word in fE or word in fV or word in fA:
                condB = (0 + 1 / denomB)
                fB[word] = condB
    i+=1
i=0   
while (i < cl.size):
    for word in new_ts[i]:
        if word in fE:
            if fE.at[0,word] > 1:
                condE = (float(fE.loc[0,word] + 1) / float(denomE))
                fE[word] = condE
        elif word in fA or word in fV or word in fB:
            condE = (0 + 1 / denomE)
            fE[word] = condE
    i+=1
i=0
while (i < cl.size):
    for word in new_ts[i]:
        if word in fV:
            if fV.at[0,word] > 1:
                condV = (float(fV.loc[0,word] + 1) / float(denomV))
                fV[word] = condV
        elif word in fA or word in fE or word in fB:
            condV = (0 + 1 / denomV)
            fV[word] = condV
    i+=1


fA.to_csv("cond_a.csv", index=False)
fB.to_csv("cond_b.csv", index=False)
fE.to_csv("cond_e.csv", index=False)
fV.to_csv("cond_v.csv", index=False)  

     
condA = pd.read_csv("cond_a.csv")
condA.head()
condB = pd.read_csv("cond_B.csv")
condB.head()
condE = pd.read_csv("cond_E.csv")
condE.head()
condV = pd.read_csv("cond_V.csv")
condV.head()

# generate classifications. 
def classify(abstracts):
    
    # Text processing
    
    print("Processing the test abstracts...")
    tst_ab = abstracts.str.split(" ")
    
    
    # Run processed abstracts through the pre-trained naive bayes classifier
    print("Classifying the test abstracts...")
    prob_classA = math.log(pA)
    prob_classB = math.log(pB)
    prob_classE = math.log(pE)
    prob_classV = math.log(pV)
    classes_array = []
    i = 0
    while (i < len(tst_ab)):
        for word in tst_ab[i]:
            if word in condA:
                prob_classA += math.log(condA.loc[0,word])
    
        for word in tst_ab[i]:
            if word in condB:
                prob_classB += math.log(condB.loc[0,word])

        for word in tst_ab[i]:
            if word in condE:
                prob_classE += math.log(condE.loc[0,word])

        for word in tst_ab[i]:
            if word in condV:
                prob_classV += math.log(condV.loc[0,word])
                
        highest = max(prob_classV, prob_classA, prob_classB, prob_classE)
        if(highest == prob_classV):
            predict = 'V'
        elif(highest == prob_classB):
            predict = 'B'
        elif(highest == prob_classA):
            predict = 'A'
        elif(highest == prob_classE):
            predict = 'E'
        i += 1
        classes_array.append(predict)
        prob_classA = math.log(pA)
        prob_classB = math.log(pB)
        prob_classE = math.log(pE)
        prob_classV = math.log(pV)
   
    return classes_array
    
    
# Load in the test set .csv
test_set = pd.read_csv("tst.csv")

# Apply the model to the test set
test_set_class_predictions = classify(test_set["abstract"])
test_set["class"] = test_set_class_predictions


# Write the test set classifications to a .csv
test_set.drop(["abstract"], axis = 1).to_csv("tst_kaggle.csv", index=False)
test_set.head()
