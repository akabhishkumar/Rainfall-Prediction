#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[85]:


raw_df=pd.read_csv("weatherAUS.csv")

raw_df.info()
raw_df #1,45,460 rows 


# In[86]:


raw_df.dropna(subset=['RainToday','RainTomorrow'],inplace = True)
raw_df.info()


# In[87]:


import plotly.express as px
import seaborn as sns


# In[88]:


fig=px.histogram(raw_df,x='Location',title='Location vs Rainy Days',color='RainToday')
fig.show()


# In[ ]:





# In[89]:


import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


px.histogram(raw_df,x='Location',title='Location vs rainy day',color='RainToday')


# In[91]:


raw_df.Location.nunique()


# In[92]:


px.histogram(raw_df,title='temp at 3pm vs rain tomorrow',x='Temp3pm',color='RainTomorrow')


# In[93]:


px.histogram(raw_df,x='RainToday',color='RainTomorrow',title='rain today vs rain tomorrow')


# In[94]:


px.scatter(raw_df.sample(2000),title='min temp vs max temp',x='MinTemp',y='MaxTemp',color='RainToday')


# In[95]:


import sklearn 
from sklearn.model_selection import train_test_split


# In[96]:


train_val_df,test_df=train_test_split(raw_df,test_size=0.2,random_state=42)
train_df,val_df=train_test_split(train_val_df,test_size=0.25,random_state=42)


# In[97]:


print(test_df.shape)
print(train_df.shape)
print(val_df.shape)


# In[98]:


year=pd.to_datetime(raw_df.Date).dt.year
train_df=raw_df[year<2015]
val_df=raw_df[year==2015]
test_df=raw_df[year>2015]


# In[99]:


input_cols=list(train_df.columns)[1:-1]
target_cols='RainTomorrow'


# In[100]:


train_inputs=train_df[input_cols].copy()
train_targets=train_df[input_cols].copy()


# In[101]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()


# In[102]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy()


# In[103]:


train_targets = train_df['RainTomorrow'].copy()
val_targets   = val_df['RainTomorrow'].copy()
test_targets  = test_df['RainTomorrow'].copy()


# In[104]:


train_targets = train_targets.map({'No':0, 'Yes':1})
val_targets   = val_targets.map({'No':0, 'Yes':1})
test_targets  = test_targets.map({'No':0, 'Yes':1})


# In[105]:


numeric_cols=train_inputs.select_dtypes(include=np.number).columns.tolist()
numeric_cols
categorical_cols=train_inputs.select_dtypes('object').columns.tolist()
categorical_cols



# In[106]:


train_inputs[numeric_cols].describe()


# In[107]:


from sklearn.impute import SimpleImputer


# 

# In[108]:


imputer=SimpleImputer(strategy='mean')
raw_df[numeric_cols].isna().sum()


# In[109]:


imputer.fit(raw_df[numeric_cols])


# In[110]:


list(imputer.statistics_)


# In[111]:


train_inputs[numeric_cols]=imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols]=imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]=imputer.transform(test_inputs[numeric_cols])


# In[112]:


from sklearn.preprocessing import MinMaxScaler


# In[113]:


scaler= MinMaxScaler()


# In[114]:


scaler.fit(raw_df[numeric_cols])


# In[115]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
train_inputs.describe()


# In[116]:


raw_df.Location.unique()
train_inputs[numeric_cols].describe()


# In[117]:


from sklearn.preprocessing import OneHotEncoder


# In[118]:


raw_df[categorical_cols].nunique()
encoder =OneHotEncoder(sparse_output=False,handle_unknown='ignore')


# In[119]:


encoder.fit(raw_df[categorical_cols])


# In[120]:


categorical_cols


# In[121]:


encoder.categories_


# In[122]:


encoded_cols=list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


# In[123]:


train_inputs[encoded_cols]=encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols]=encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols]=encoder.transform(test_inputs[categorical_cols])


# In[124]:


test_inputs


# In[125]:


# training the model


# In[126]:


from sklearn.linear_model import LogisticRegression


# In[127]:


model= LogisticRegression(solver='liblinear', class_weight='balanced')


# In[128]:


model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# In[129]:


print(model.coef_.tolist())


# In[130]:


val_preds = model.predict(val_inputs[numeric_cols + encoded_cols])


# In[131]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_targets, val_preds)
print("Validation Accuracy:", accuracy)


# In[132]:


from sklearn.metrics import classification_report

print(classification_report(val_targets, val_preds))


# In[133]:


x_train=train_inputs[numeric_cols+encoded_cols]
x_val=val_inputs[numeric_cols+encoded_cols]
x_test=test_inputs[numeric_cols+encoded_cols]


# In[134]:


train_preds=model.predict(x_train)
train_preds


# In[135]:


train_targets


# In[136]:


train_probs=model.predict_proba(x_train)
train_probs


# In[137]:


from sklearn.metrics import confusion_matrix


# In[138]:


confusion_matrix(train_targets,train_preds,normalize='true')


# In[139]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));

    return preds


# In[140]:


train_preds = predict_and_plot(x_train, train_targets, 'Training')


# In[141]:


val_preds = predict_and_plot(x_val, val_targets, 'Validatiaon')


# In[142]:


new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[143]:


new_input_df = pd.DataFrame([new_input])
new_input_df


# In[144]:


new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


# In[145]:


X_new_input = new_input_df[numeric_cols + encoded_cols]
X_new_input


# In[146]:


prediction = model.predict(X_new_input)[0]
if prediction == 1:
    print("Yes")
else:
    print("No")


# In[147]:


prob= model.predict_proba(X_new_input)[0]
prob


# In[148]:


import joblib


# In[149]:


aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[150]:


joblib.dump(aussie_rain, 'aussie_rain.joblib')


# In[151]:


aussie_rain2 = joblib.load('aussie_rain.joblib')


# In[152]:


test_preds2 = aussie_rain2['model'].predict(x_test)
accuracy_score(test_targets, test_preds2)


# In[153]:


jupyter nbconvert --to script LogisticRegression.ipynb


# In[ ]:




