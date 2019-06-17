# Project Title: Grab AI Challenge - Safety

### Problem Statement
Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips. （https://www.aiforsea.com/safety）

### Approach
Dangerous driving behaviour includes 
1. speeding
2. harsh breaking
3. heavy acceleration
4. dangerous cornering
5. swerving, etc

The task is to identify if above dangerous driving behaviour has happened, by examine patterns in mobile phone accelerometer, gyroscope and GPS data for each booking. Per second data of each booking is provided

Traditionally, this was done through manually set thredhold. However, Convolutional Neural Networs (CNN) is superior at identifying deep patterns for multiple long time series data.

This study has transform the feature data into a 3D array: (#bookings, #seconds, #features)， and then trains a 1D CNN (Resnet) on it. Best model is evaluated with AUC ROC.

For the purpose of this study, long trips are broken down into several 1-hour sub-trips. Final predicted probability takes the maximum value of the predicted probability of all sub-trips. For real world application, prediction can be further broken down into 5/10 min short intervals. Grab can use prediction result to give warning notifications to drivers that are driving dangerously.

### Prerequisits

1. install anaconda and jupyter notebook
2. install Python (version 3.6.7)
3. (conda) install the following python packages
	- pandas
	- sklearn
	- joblib
	- matplotlib
	- numpy
	- keras
	- tensorflow-gpu

## Running the prediction code on test data

1. place test data (in csv) in the `data/safety/test` folder
2. cd to the project folder grab_ai_challenge
3. run the jupyter noteboook `notebook/test_data_preprocessing.ipynb` line by line
4. run the jupyter noteboook `notebook/test_1D_CNN.ipynb` line by line
5. prediction output is written to `data/safety/prediction_output.csv`
6. prediction output csv has two columns - bookingID and predict_prob, ordered by bookingID. This can then be used to evaluate AUC ROC against true labels.

