import numpy as np  
import pandas as pd  
import scipy.stats as stats  
import matplotlib.pyplot as plt  
import sklearn



#Python method for handling nan values
#Numerical columns replaced with mean
#String columns replaced with mode
from collections import Counter
def dataCleaning(inputData,features,uniqueId,target):
    updatedData =  inputData[target]   
    dataForClean=pd.DataFrame()
    for i in range(0,len(features)):
        dataForClean=pd.concat([dataForClean,pd.DataFrame(inputData[features[i]])],axis=1)
    dataForClean=pd.concat([dataForClean,pd.DataFrame(inputData[target])],axis=1)
    inputData=dataForClean
    for i in range(0,len(inputData.columns)):
        colName=inputData.columns[i]
        if( (inputData[colName].dtype.name=="object" and inputData[colName].dtype.name!= target) and colName != uniqueId):
            mode=Counter(inputData[colName])
            modeval= mode.most_common(1)[0][0]
            inputData[colName]= inputData[colName].replace(np.NaN, modeval)
        else:
            inputData[colName]= inputData[colName].replace(np.NaN, np.mean(inputData[colName]))

    
    return inputData,updatedData




#Python method for pre-processing the data
#Categorical variables are label and one hot encoded
#Scale the data using standard scaler method in sklearn
def dataPreProcess(inputData,target):
    from sklearn import preprocessing 
    labelencoder= preprocessing.LabelEncoder()
    onehotencoder = preprocessing.OneHotEncoder(sparse=False)
    notencoded = pd.DataFrame()
    encoded = pd.DataFrame()
    colNames=[]
    for i in range(0,len(inputData.columns)):
        colNames.append(inputData.columns[i])
        
    for i in range(0,len(colNames)):
        colName=colNames[i]
        if( (inputData[colName].dtype.name=="object" and inputData[colName].dtype.name!= target)):
            df =  labelencoder.fit_transform(inputData[colName])
            df = np.array(inputData[colName])
            df =   df.reshape(len(df),1)
            onehot_encoded = onehotencoder.fit_transform(df)
            onehot_encoded = pd.DataFrame(onehot_encoded)
            onehot_encoded = onehot_encoded.drop(len(onehot_encoded.columns)-1,axis=1)
            encoded=pd.concat([onehot_encoded,encoded],axis=1)       
        else:
            notencoded=  pd.concat([notencoded,inputData[colName]],axis=1)
    combinedData=pd.concat([encoded,notencoded],axis=1)
    updatedData=pd.DataFrame()
    updatedData=pd.concat([updatedData,combinedData],axis=1) 
    Y = pd.DataFrame() 
    Y = pd.DataFrame(updatedData[target])
    del updatedData[target]
    X = pd.DataFrame()
    X = updatedData
    
    #.............................Data Scaled......................#
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
   
    return X,Y



def datasplit(X,Y,size):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=size, shuffle=False)
    return X_train, X_test, Y_train, Y_test




def decsiontreeClassification(X_train, X_test, Y_train, Y_test):
    from sklearn import tree
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm


def logisticRegression(X_train, X_test, Y_train, Y_test):
    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression(random_state = 0)
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm



def xgboostClassification(X_train, X_test, Y_train, Y_test):
    from xgboost import XGBClassifier
    regressor = XGBClassifier()
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm

def randomForestClassification(X_train, X_test, Y_train, Y_test):
    from sklearn.ensemble import RandomForestClassifier
    regressor = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    regressor.fit(X_train, Y_train)
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm


def svmClassification(X_train, X_test, Y_train, Y_test):
    from sklearn.svm import SVC
    regressor = SVC(kernel = 'linear', random_state = 0)
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm


def knnClassification(X_train, X_test, Y_train, Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    regressor = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    regressor.fit(X_train,Y_train)        
    predictions = np.ceil(regressor.predict(X_test))
    from sklearn import metrics       
    cm = metrics.confusion_matrix(Y_test, predictions)
    return predictions,cm


def execute(data,target):
    print("Entered execute")
    data = data
    tempdata = data
    target = target

    data = data.select_dtypes(include=[np.number])
    uniqueId = ""
    testsize = 0.3
    features = np.array(data.columns)
    features = features[features != target]



    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(tempdata[target])
    values=np.array(le.transform(tempdata[target]))
    invalues=list(le.inverse_transform([0,1]))
    dataframe = pd.DataFrame(values,columns=[target])
    data[target] = dataframe[target].values
    print(data.head(10))

    invdict= {0: invalues[0],
    1: invalues[1]
    }



    if(uniqueId==""):
        uniqueId=uniqueId
        uniqueIdGenerator=np.arange(1,len(data))
        UqId=pd.DataFrame(list(uniqueIdGenerator),columns=["UniqueId"])
        initData=pd.concat([UqId,data],axis=1) 
        uniqueId=initData.columns[0]


    data,updatedData=dataCleaning(data,features,uniqueId,target)
    X,Y=dataPreProcess(data,target)
    X_train, X_test, Y_train, Y_test=datasplit(X,Y,testsize)

    #Decision tree classification
    predictions,cm=decsiontreeClassification(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('decisiontreeClassification.csv', index = False)
    daccuracy=accuracy
    dprecision=precision
    drecall=recall
    df1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)

    #logistic regression
    predictions,cm=logisticRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('logisticRegression.csv', index = False)
    laccuracy=accuracy
    lprecision=precision
    lrecall=recall
    lf1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)

    #xgboost classification
    predictions,cm=xgboostClassification(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('xgboostClassification.csv', index = False)
    xaccuracy=accuracy
    xprecision=precision
    xrecall=recall
    xf1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)

    #randomForestClassification
    predictions,cm=randomForestClassification(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('randomForestClassification.csv', index = False)
    raccuracy=accuracy
    rprecision=precision
    rrecall=recall
    rf1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)


    #svmClassification
    predictions,cm=svmClassification(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('svmClassification.csv', index = False)
    saccuracy=accuracy
    sprecision=precision
    srecall=recall
    sf1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)


    

    #knnClassification
    predictions,cm=knnClassification(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=initData[uniqueId][len(X_train):len(data)]
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    prednactualfinal=pd.concat([testStart,prednactual],axis=1)
    truePositive = cm[0,0]
    falsePositive= cm[1,0]
    trueNegative= cm[1,1]
    falseNegative=cm[0,1]
    accuracy= (truePositive + trueNegative) / (truePositive+falsePositive+trueNegative+falseNegative)
    precision= (truePositive) /(truePositive + falsePositive)
    recall=   (truePositive) /(truePositive+falseNegative)
    f1score= (2*precision*recall) / (precision+recall)
    prednactualfinal.to_csv('knnClassification.csv', index = False)
    kaccuracy=accuracy
    kprecision=precision
    krecall=recall
    kf1score=f1score
    print("skdskdsd")
    print(daccuracy)
    print(dprecision)
    print(drecall)
    array =[daccuracy,dprecision,drecall,df1score,laccuracy,lprecision,lrecall,lf1score,xaccuracy,xprecision,xrecall,xf1score,raccuracy,rprecision,rrecall,rf1score,saccuracy,sprecision,srecall,sf1score,kaccuracy,kprecision,krecall,kf1score,invdict]
    


    return array
