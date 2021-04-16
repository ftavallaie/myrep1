#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# In[31]:


st.write('''
# STOCK PREDICTOR

**By Farshad Tavallaie**

''')

img = Image.open("BUGomv.jpg")
st.image(img, width=600)


# In[32]:


st.sidebar.header("Enter data")
def data():
    symbol = st.sidebar.selectbox("Select the symbol", ["AMAZON", "FIROOZE", "FOOLAD", "KHALIJ", "KHODRO", "TAPPICO", "TESLA"])
    n = st.sidebar.text_input("Number of days to be predicted", 5)
    return n, symbol


# In[33]:


def get_data():

    if (symbol == "AMAZON"):
        df = pd.read_csv("AMZN.csv")
    elif (symbol == "FIROOZE"):
        df = pd.read_csv("Firooze ETF-a.csv")

        # I added this line for data received by app
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    elif (symbol == "FOOLAD"):
        df = pd.read_csv("S Mobarakeh Steel-a.csv")
        
        # I added this line for data received by app
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    elif (symbol == "KHALIJ"):
        df = pd.read_csv("Khalij Fars-a.csv")

        # I added this line for data received by app
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        
    elif (symbol == "KHODRO"):
        df = pd.read_csv("Iran Khodro-a.csv")
        
        # I added this line for data received by app
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    elif (symbol == "TAPPICO"):
        df = pd.read_csv("Tamin Petro.-a.csv")

        # I added this line for data received by app
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    elif (symbol == "TESLA"):
        df = pd.read_csv("TSLA.csv")
        
    df = df.set_index(pd.DatetimeIndex(df["Date"].values))
    return df



# In[34]:


def get_company_name(symbol):
    if symbol == "AMAZON":
        return "Amazon.com"
    elif symbol == "FIROOZE":
        return "Firooze ETF"
    elif symbol == "FOOLAD":
        return "S*Mobarakeh Steel"
    elif symbol == "KHALIJ":
        return "Khalij Fars"
    elif symbol == "KHODRO":
        return "Iran Khodro"
    elif symbol == "TAPPICO":
        return "Tamin Petro."
    elif symbol == "TESLA":
        return "Tesla"
    else:
        return "NONE"


# In[35]:


n, symbol = data()
df = get_data()
company = get_company_name(symbol)
st.header(company + " Adj Close Price\n")
st.line_chart(df["Adj Close"])
st.header(company + " Volume\n")
st.line_chart(df["Volume"])
st.header("Stock Data")
st.write(df.describe())


# In[36]:


df = df[["Adj Close"]]
forecast = int(n)
df["Prediction"] = df[["Adj Close"]].shift(-forecast)
x = np.array(df.drop(["Prediction"], 1))
x = x[:-forecast]
y = np.array(df["Prediction"])
y = y[:-forecast]


# In[37]:


xtrain , xtest , ytrain , ytest = train_test_split(x, y, test_size=0.2)
mysvr = SVR(kernel="rbf", C=1000, gamma=0.1)
mysvr.fit(xtrain, ytrain)
svrconf = mysvr.score(xtest, ytest)
st.header("SVR Accuracy")
st.success(svrconf)


# In[38]:


x_forecast = np.array(df.drop(["Prediction"], 1))[-forecast:]
svrpred = mysvr.predict(x_forecast)
st.header("SVR Prediction")
st.success(svrpred)


# In[39]:


lr = LinearRegression()
lr.fit(xtrain, ytrain)
lrconf = lr.score(xtest, ytest)
st.header("LR Accuracy")
st.success(lrconf)


# In[40]:


lrpred = lr.predict(x_forecast)
st.header("LR Prediction")
st.success(lrpred)


# In[ ]:

plt.style.use("fivethirtyeight")

if (svrconf > lrconf):
    st.header(f"Prediction chart based on SVR\n")
    theLast = round(svrpred[len(svrpred) - 1], 2)
    theMean = round(np.mean(svrpred), 2)
    theIncrease1 = round(float((svrpred[int(n) - 1] - df.tail(1)["Adj Close"]) / df.tail(1)["Adj Close"] * 100), 2)
    theIncrease2 = round(float((theMean - df.tail(1)["Adj Close"]) / df.tail(1)["Adj Close"] * 100), 2)

    myArray = np.array(range(0, int(n) + 1)).reshape(int(n)+1, 1)
    svrpred = np.hstack((df.tail(1)["Adj Close"], svrpred))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.legend("Price")
    ax.plot(myArray, svrpred)
    ax.plot(myArray, np.full((int(n)+1,), theMean), linestyle="dotted")
    plt.legend(["Price", "Mean"])
    ax.set_xlabel("Day No.")
    # ax.set_ylabel("Price")

    st.write(fig)

    st.write("Dataset updated in 2021-04-14")
    st.write(f"The last predicted price: {theLast} ({theIncrease1})")
    st.write(f"The predicted mean price: {theMean} ({theIncrease2})")

else:
    st.header(f"Prediction chart based on LR\n")
    theLast = round(lrpred[len(lrpred) - 1], 2)
    theMean = round(np.mean(lrpred), 2)
    theIncrease1 = round(float((lrpred[int(n) - 1] - df.tail(1)["Adj Close"]) / df.tail(1)["Adj Close"] * 100), 2)
    theIncrease2 = round(float((theMean - df.tail(1)["Adj Close"]) / df.tail(1)["Adj Close"] * 100), 2)

    myArray = np.array(range(0, int(n) + 1)).reshape(int(n)+1, 1)
    lrpred = np.hstack((df.tail(1)["Adj Close"], lrpred))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.legend("Price")
    ax.plot(myArray, lrpred)
    ax.plot(myArray, np.full((int(n)+1,), theMean), linestyle="dotted")
    plt.legend(["Price", "Mean"])
    ax.set_xlabel("Day No.")
    # ax.set_ylabel("Price")

    st.write(fig)

    st.write("Dataset updated in 2021-04-14")
    st.write(f"The last predicted price: {theLast} (%{theIncrease1})")
    st.write(f"The predicted mean price: {theMean} (%{theIncrease2})")






