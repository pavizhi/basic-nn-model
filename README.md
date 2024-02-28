## Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.This includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.

## Neural Network Model

The provided description outlines a neural network architecture with four hidden layers. These hidden layers utilize ReLU activation functions and have 4 and 4 neurons, respectively. Following these hidden layers is a single-neuron output layer with a linear activation function. This network takes a single variable as input and is designed to predict continuous outputs.
![](./d.png)
## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:B.Pavizhi
### Register Number:212221230077
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('EX-1').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df.head(20)

x=df[['INPUT']].values
y=df[['OUTPUT']].values
x
  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scalar=MinMaxScaler()
scalar.fit(x_train)
x_train1=scalar.transform(x_train)
ai=Sequential([Dense (units = 4, activation = 'relu', input_shape = [1]), Dense (units = 4, activation = 'relu'),Dense (units = 1)])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(x_train1,y_train,epochs=10000)
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
X_test1 = scalar.transform(x_test)
ai.evaluate(X_test1,y_test)
X_n1 = [[10]]
X_n1_1 = scalar.transform(X_n1)
ai.predict(X_n1_1)
 
```
## Dataset Information
![](./1.png)

## OUTPUT
## Epoch Training
![](./2.png)
## Training Loss Vs Iteration Plot
![](./3.png)

## Test Data Root Mean Squared Error
![](./4.png)


## New Sample Data Prediction
![](./5.png)


## RESULT
Thus a basic neural network regression model for the given dataset is written and executed successfully.
