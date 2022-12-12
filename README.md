# Stock-Prediction
This repository uses Generative Models to predict stock prices using time series data from the past. We used HMMs, VAEs and GANs to predict the closing price of the stock in the future, to assist traders make decisions. The data is taken from https://github.com/eliangcs/pystock-data

HMM: The HMM uses the hmmlearn python package to learn a Gaussian HMM on stock data. The model can be run using the hmm.py file. By default it plots the predictions for NASDAQ from Jan 2018 to May 2018. The plots for the model selection using AIC and BIC can be generated using the AIC_model_selection.py file

GAN: We used TensorFlow to implement a TimeGAN which can be found in the TimeGAN.py. The file generates the necesary plots that match the report.

