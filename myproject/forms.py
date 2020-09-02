from django import forms  
class modelForm(forms.Form):  
    targetname = forms.CharField(label="Enter the name of target variable",max_length=50)  
    