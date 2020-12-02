### STOCK PRICE PREDICTION USING LSTM
# Author: Saarthak Srivastava
# Keras and Tensorflow 2.3

# Data Collection
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd
# We can add any stock to get desired ticker live data
stock='^BSESN'
today = date.today()
# We can get data by our choice by giving days bracket
start_date= '2017-06-01'
files=[]
print (stock)
data = pdr.get_data_yahoo(stock, start=start_date, end=today)
dataname= stock+'_'+str(today)
files.append(dataname)
# Create a data folder in your current dir.
data.to_csv('./data/'+dataname+'.csv')
df= pd.read_csv('./data/'+ str(files[0])+'.csv')
df1=df.reset_index()['Close']

# Plotting historical chart
import matplotlib.pyplot as plt
plt.figure(figsize = (18,9))
plt.plot(range(df1.shape[0]),(df1))
plt.grid(color='b', ls = '-.', lw = 0.25)
plt.xticks(range(0,df.shape[0],100),df['Date'].loc[::100],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (INR)',fontsize=18)
plt.show()

# LSTMs are very sensetive to scale of data so we'll apply MinMax Scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.7)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----49   50 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 50
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is reqd for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

# Training the model
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,
            batch_size=32,verbose=1)

### Lets Do the prediction
import tensorflow as tf
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transform back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2):len(df1), :] = test_predict
# plot baseline and predictions
plt.figure(figsize = (18,9))
plt.plot(scaler.inverse_transform(df1),color='b',label='True')
plt.plot(trainPredictPlot,color='yellow',label='Train')
plt.plot(testPredictPlot,color='green', label='Prediction')
plt.grid(color='b', ls = '-.', lw = 0.25)
plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend(fontsize=18)
plt.show()

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp2_input=temp_input

# demonstrate prediction for next 30 days
from numpy import array
lst_output=[]
n_steps=x_input.shape[1]
i=0
while(i<30):
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        mult=temp2_input[-1]/yhat[0]
        if(i%3==0):
            temp_input.extend((yhat[0])*mult.tolist())
        elif(i%10==0):
            temp_input.extend((yhat[0]).tolist())
        else:
            temp_input.extend(np.array([temp_input[-1]]).tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i+=1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        mult=temp2_input[-1]/yhat[0]
        temp_input.extend((yhat[0]*mult).tolist())
        lst_output.extend(yhat.tolist())
        i+=1

day_new=np.arange(-49,-49+time_step)
day_pred=np.arange(-49+time_step,-19+time_step)

plt.figure(figsize = (18,9))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-time_step:]),
            label='Past')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label='Future')
plt.grid(color='b', ls = '-.', lw = 0.25)
plt.xlabel('Day (0 means today)')
plt.ylabel('Price')
plt.legend(fontsize=18)
plt.show()

print('Predicted stock price after 30 Days: ',
        scaler.inverse_transform(lst_output)[-1])

df3=df1.tolist()
df3.extend(lst_output)
df3=scaler.inverse_transform(df3).tolist()
plt.figure(figsize = (18,9))
plt.plot(df3)
plt.grid(color='b', ls = '-.', lw = 0.25)
plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing Price (INR)',fontsize=18)
plt.show()

#####################################