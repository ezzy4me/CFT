from re import X
import pandas as pd
import manual_tree as mt
from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import os
path = os.getcwd()

#=============Abstract======================
"""
STEP
0. load dataset
1. making user own manual decision tree.
2. there will be input so we should retrun output.
3. assemble each part
"""
#===========================================



#===========================================
#STEP.0
#load train dataset
cancer = load_breast_cancer()
df1 = pd.DataFrame(cancer['data'], columns=['v'+str(i) for i in range(30)])
df2 = pd.DataFrame(cancer['target'], columns=['Y'])
train = pd.concat([df1, df2], axis=1)
valid = pd.concat([df1, df2], axis=1)
#===========================================



#===========================================
#STEP.1
#Model Hyperparameter
save_path = '/Users/min/Documents/GitHub' #ur pathway
model_type = 'PD'      
y_label = 'Y'          
criterion = 'entropy'      # decision tree bifurcation criteria, gini entropy
min_samples_leaf = 1  # minimum sample size of leaf node
logger=''               # log information

#demo create tree
manaul_obj = mt.Manaul(train, valid, y_label, save_path, model_type, criterion, min_samples_leaf, logger)
#============================================



#===========================================
#STEP.2
# generate DecicionTreeClassifier 
# dt_clf = DecisionTreeClassifier(random_state=156)

# load dataset from step0, seperate train and test set

# X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2, random_state=11)
# DecisionTreeClassifier 학습
# dt_clf.fit(X_train, y_train)
#===========================================



#===========================================
#STEP.3
#prototype modeling
def prototype(row):

    while True:
        #generate DecicionTreeClassifier 
        dt_clf = DecisionTreeClassifier(random_state=156)    

        var_list = []    
        for i in range(3):

            manaul_obj.get_pool_node_id(i)  # divide the data into two pools i and i+1 according to the feature V(text)
            text = input('variable ex)v13 :  ')


            value = float(input('split_value  ex)14 :  '))
            manaul_obj.calculate_feature_split(variable_selected = text, split_value = value) #v20
            manaul_obj.save_step_split()
            print('\n')
            var_list.append(text)

        # load data
        X_train = df1[var_list]
        y_train = df2
        # train model for d_tree
        dt_clf.fit(X_train, y_train)
        # return ouput
        input_x = X_train.iloc[row] #select rows for inferencing
        input_x = pd.DataFrame(input_x) #reshape for acceptable input
        input_reshape_x = input_x.transpose() #(1, number of feature)
           
        output_y = dt_clf.predict(input_reshape_x)
        print(output_y)

        return output_y
        
#============================================



prototype(40)
