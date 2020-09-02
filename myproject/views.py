from django.shortcuts import render  
from django.template import loader  
from django.http import HttpResponseBadRequest, HttpResponse
import pandas as pd
import numpy as np
import requests
from django.shortcuts import render  
from myproject.forms import modelForm  

#Create your views here  
def hello(request):  
    return HttpResponse("<h2>Hello, Welcome to Django!</h2>")  

def eda(df):
    data = df
    from pandas_profiling import ProfileReport
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file("your_report.html")

def logisticregression():
    data = score,precision,recall,f1score,prednactualfinal
    import logisticregression



def outputeda(request):
    print('EDA')
    if request.method == 'POST':
        import sample
        data= sample.readfile()
        #return render(request,'hello.html')
        eda(data)
        print(data)
        return render(request,"outputeda.html") 


def outputmodelregression(request):
    print('REGRESSION')
    if request.method == 'POST':
        modelobj = modelForm() 
        print(modelobj)
        import sample
        data= sample.readfile()
        form = modelForm(request.POST)
        target = form['targetname'].value()
        print("target from view : ",target)
        import myproject.regression
        array  = myproject.regression.execute(data,target)
        #array = np.array([score,adjr,mae,mse,rmse])
        print("..............Array..............")
        print(array)
        print(array[0])
        #df = pd.DataFrame(array,index=["RSqaured","AdjustedRSqaured"])
        #return HttpResponse(df.to_html())
        regdict = {
        'rRSquared': array[0], 
        'rAdjRSQuared': array[1],
        'rMAE':array[2],
        'rMSE':array[3],
        'rRMSE':array[4],

        'lRSquared':array[5],
        'lAdjRSQuared':array[6],
        'lMAE':array[7],
        'lMSE':array[8],
        'lRMSE':array[9],

        
        'dRSquared':array[10],
        'dAdjRSQuared':array[11],
        'dMAE':array[12],
        'dMSE':array[13],
        'dRMSE':array[14],

        
        'sRSquared':array[15],
        'sAdjRSQuared':array[16],
        'sMAE':array[17],
        'sMSE':array[18],
        'sRMSE':array[19],

        'xRSquared':array[20],
        'xAdjRSQuared':array[21],
        'xMAE':array[22],
        'xMSE':array[23],
        'xRMSE':array[24]
        }

        maxval = max(regdict, key=regdict.get)
        maxdict = {'maxval':maxval}

        maxval = np.array([array[1],array[6],array[11],array[16],array[21]])
        maxElement = np.amax(maxval)
        algo= list(regdict.keys())[list(regdict.values()).index(maxElement)]
        print("BoNp")
        print(algo)
        algodict={"d":"Decision tree regression",
        "l":"Multiple linear regression",
        "x":"XGBoost regression",
        "r": "Random forest regression",
        "s": "Support vector regression"
        }

        prefix = algo[0]
        print(prefix)
        algoname= algodict["r"]
        print(algoname)
        regdict["max"] = algoname


        return render(request,"outputmodelreg.html",regdict)



def outputmodelclassification(request):
    print('CLASSIFICATION')
    if request.method == 'POST':
        modelobj = modelForm() 
        print(modelobj)
        import sample
        data= sample.readfile()
        form = modelForm(request.POST)
        target = form['targetname'].value()
        print("target from view : ",target)
        import myproject.classification
        array  = myproject.classification.execute(data,target)
        print("..............Array..............")
        print(array)
        classdict= {
        'dAccuracy': array[0], 
        'dPrecision': array[1],
        'dRecall':array[2],
        'dF1score':array[3],

        'lAccuracy': array[4], 
        'lPrecision': array[5],
        'lRecall':array[6],
        'lF1score':array[7],


        'xAccuracy': array[8], 
        'xPrecision': array[9],
        'xRecall':array[10],
        'xF1score':array[11],


        'rAccuracy': array[12], 
        'rPrecision': array[13],
        'rRecall':array[14],
        'rF1score':array[15],


        'sAccuracy': array[16], 
        'sPrecision': array[17],
        'sRecall':array[18],
        'sF1score':array[19],


        'kAccuracy': array[20], 
        'kPrecision': array[21],
        'kRecall':array[22],
        'kF1score':array[23]
        }
        


        maxval = max(classdict, key=classdict.get)
        maxdict = {'maxval':maxval}

        maxval = np.array([array[3],array[7],array[11],array[15],array[19],array[23]])
        maxElement = np.amax(maxval)
        algo= list(classdict.keys())[list(classdict.values()).index(maxElement)]
        print("BoNp")
        print(algo)
        algodict={"d":"Decision tree classification",
        "l":"Logistic classification",
        "x":"XGBoost classification",
        "r": "Random forest classification",
        "s": "Support vector machine",
        "k":"K Nearest neighbour classification"}

        prefix = algo[0]
        print(prefix)
        algoname= algodict["r"]
        print(algoname)
        classdict["max"] = algoname
        indict=array[24]

        print("Dicagjsjka")
        print(indict)
        print(indict[0], indict[1])
        classdict["class0"] = indict[0]
        classdict["class1"] = indict[1]

      
        return render(request,"outputmodelclass.html",classdict)



def index(request):
    return render(request,"index.html")  





