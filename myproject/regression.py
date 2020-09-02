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



#Python method for performing logistic regression
def randomForestRegression(X_train, X_test, Y_train, Y_test): 
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestRegressor
    RFregr = RandomForestRegressor(max_depth = 10,random_state = 0,n_estimators = 100)
    RFregr.fit(X_train,Y_train)
    predictions = RFregr.predict(X_test) 
    score = RFregr.score(X_test, Y_test)
    return predictions,score

def decisionTreeRegression(X_train, X_test, Y_train, Y_test): 
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train,Y_train)        
    predictions = regressor.predict(X_test) 
    score = regressor.score(X_test, Y_test)
    return predictions,score

def supporVectorRegression(X_train, X_test, Y_train, Y_test): 
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train,Y_train)        
    predictions = regressor.predict(X_test) 
    score = regressor.score(X_test, Y_test)
    return predictions,score
    
def linearRegression(X_train, X_test, Y_train, Y_test): 
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)        
    predictions = regressor.predict(X_test) 
    score = regressor.score(X_test, Y_test)
    return predictions,score

def execute(data,target):
    data = data
    target = target
    data = data.select_dtypes(include=[np.number])
    features = np.array(data.columns)
    features = features[features != target]
    uniqueId = ""
    testsize = 0.3

    
 

    if(uniqueId==""):
        uniqueId=uniqueId
        uniqueIdGenerator=np.arange(1,len(data))
        UqId=pd.DataFrame(list(uniqueIdGenerator),columns=["UniqueId"])
        initData=pd.concat([UqId,data],axis=1) 
        uniqueId=initData.columns[0]


    data,updatedData=dataCleaning(data,features,uniqueId,target)
    X,Y=dataPreProcess(data,target)
    X_train, X_test, Y_train, Y_test=datasplit(X,Y,testsize)

    #RandomForest
    predictions,score=randomForestRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    adjr= 1-(1-score)*(n-1)/(n-p-1)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=list(initData[uniqueId][len(X_train):len(data)])
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    rprednactualfinal=pd.concat([testStart,prednactual],axis=1)
    from sklearn import metrics
    print("RSquared: ",score)
    print("AdjustedRSquared: ",adjr)
    print('MAE', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE', metrics.mean_squared_error(Y_test, predictions))
    print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    MAE = metrics.mean_absolute_error(Y_test, predictions)
    MSE =  metrics.mean_squared_error(Y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    rscore = score
    radjr = adjr
    rMAE = MAE
    rMSE = MSE
    rRMSE = RMSE
    rprednactualfinal.to_csv('RandomForestRegression.csv', index = False)


    #linear regression
    predictions,score=linearRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    adjr= 1-(1-score)*(n-1)/(n-p-1)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=list(initData[uniqueId][len(X_train):len(data)])
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    lprednactualfinal=pd.concat([testStart,prednactual],axis=1)
    from sklearn import metrics
    print("RSquared: ",score)
    print("AdjustedRSquared: ",adjr)
    print('MAE', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE', metrics.mean_squared_error(Y_test, predictions))
    print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    MAE = metrics.mean_absolute_error(Y_test, predictions)
    MSE =  metrics.mean_squared_error(Y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    lscore = score
    ladjr = adjr
    lMAE = MAE
    lMSE = MSE
    lRMSE = RMSE
    lprednactualfinal.to_csv('LinearRegression.csv', index = False)

    #DescionTree
    predictions,score=decisionTreeRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    adjr= 1-(1-score)*(n-1)/(n-p-1)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=list(initData[uniqueId][len(X_train):len(data)])
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    dprednactualfinal=pd.concat([testStart,prednactual],axis=1)
    from sklearn import metrics
    print("RSquared: ",score)
    print("AdjustedRSquared: ",adjr)
    print('MAE', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE', metrics.mean_squared_error(Y_test, predictions))
    print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    MAE = metrics.mean_absolute_error(Y_test, predictions)
    MSE =  metrics.mean_squared_error(Y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    dscore = score
    dadjr = adjr
    dMAE = MAE
    dMSE = MSE
    dRMSE = RMSE
    dprednactualfinal.to_csv('DecisionTreeRegression.csv', index = False)


    
    #SupportVectorRegression
    predictions,score=supporVectorRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    adjr= 1-(1-score)*(n-1)/(n-p-1)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=list(initData[uniqueId][len(X_train):len(data)])
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    sprednactualfinal=pd.concat([testStart,prednactual],axis=1)
    from sklearn import metrics
    print("RSquared: ",score)
    print("AdjustedRSquared: ",adjr)
    print('MAE', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE', metrics.mean_squared_error(Y_test, predictions))
    print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    MAE = metrics.mean_absolute_error(Y_test, predictions)
    MSE =  metrics.mean_squared_error(Y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    sscore = score
    sadjr = adjr
    sMAE = MAE
    sMSE = MSE
    sRMSE = RMSE
    sprednactualfinal.to_csv('SupportVectorRegression.csv', index = False)

     
    #XGBoost
    predictions,score=supporVectorRegression(X_train, X_test, Y_train, Y_test)
    n=len(data)
    p=len(features)
    adjr= 1-(1-score)*(n-1)/(n-p-1)
    output1=pd.DataFrame(list(Y_test[target]),columns=["Actual"])
    output2=pd.DataFrame(list(predictions),columns=["Predicted"])
    prednactual=pd.concat([output1,output2],axis=1)
    testStart=list(initData[uniqueId][len(X_train):len(data)])
    testStart=pd.DataFrame(testStart,columns=[uniqueId])
    testStart.reset_index(drop=True, inplace=True)
    prednactual.reset_index(drop=True, inplace=True)
    xprednactualfinal=pd.concat([testStart,prednactual],axis=1)
    from sklearn import metrics
    print("RSquared: ",score)
    print("AdjustedRSquared: ",adjr)
    print('MAE', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE', metrics.mean_squared_error(Y_test, predictions))
    print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    MAE = metrics.mean_absolute_error(Y_test, predictions)
    MSE =  metrics.mean_squared_error(Y_test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    xscore = score
    xadjr = adjr
    xMAE = MAE
    xMSE = MSE
    xRMSE = RMSE

    array =np.array([rscore,radjr,rMAE,rMSE,rRMSE,lscore,ladjr,lMAE,lMSE,lRMSE,dscore,dadjr,dMAE,dMSE,dRMSE,sscore,sadjr,sMAE,sMSE,sRMSE,xscore,xadjr,xMAE,xMSE,xRMSE])
    xprednactualfinal.to_csv('xgboost.csv', index = False)

    

    return array
