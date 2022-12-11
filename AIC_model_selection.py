# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:51:28 2022

@author: ssnaik
"""

import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
#plt.style.use('ggplot')
 
np.random.seed(1)
class StockPredictor(object):
    def __init__(self, company, test_size=0.1,
                 n_hidden_states=4, n_latency_days=1,
                 n_steps_frac_change=20, n_steps_frac_high=10,
                 n_steps_frac_low=10):
        self._init_logger()
 
        self.company = company
        self.n_latency_days = n_latency_days
 
        self.hmm = GaussianHMM(n_components=n_hidden_states)
 
        self._split_train_test_data(test_size)
 
        self._compute_all_possible_outcomes(
            n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
 
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
 
    def _split_train_test_data(self, test_size):
        data = pd.read_csv(
            'data/company_data/{company}.csv'.format(company=self.company))
        _train_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=False)
 
        self._train_data = _train_data
        self._test_data = test_data
 
    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
 
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
 
        return np.column_stack((frac_change, frac_high, frac_low))
 
    def fit(self):
        self._logger.info('>>> Extracting Features')
        feature_vector = StockPredictor._extract_features(self._train_data)
        self._logger.info('Features extraction Completed <<<')
            
        self.hmm.fit(feature_vector)
 
    def _compute_all_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
 
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))
 
    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = StockPredictor._extract_features(
            previous_data)
 
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]
 
        return most_probable_outcome
 
    def predict_close_price(self, day_index):
        open_price = self._test_data.iloc[day_index]['open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(
            day_index)
        return open_price * (1 + predicted_frac_change)
 
    def predict_close_prices_for_days(self, days, with_plot=False):
        predicted_close_prices = []
        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
 
        if with_plot:
            
            test_data = self._test_data[0: days]
            days = np.array(test_data['date'], dtype="datetime64[ms]")
            actual_close_prices = test_data['close']
 
            fig = plt.figure()
 
            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', label="actual")
            axes.plot(days, predicted_close_prices, 'r+-', label="predicted")
            axes.set_title('{company}'.format(company=self.company))
 
            fig.autofmt_xdate()
 
            plt.legend()
            plt.show()
 
        return predicted_close_prices, days, actual_close_prices
    
    def aic_cal(self,data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
        
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / close_price
        frac_high = (high_price - open_price) / close_price
        frac_low = (open_price - low_price) / close_price
        
        #define response variable
        y = close_price
        
        #define predictor variables
        x = np.array([frac_change, frac_high, frac_low]).transpose()
        
        
        # #add constant to predictor variables
        # x = sm.add_constant(x)
        
        # #fit regression model
        # model = sm.OLS(y, x).fit()
        
        
    def bic_cal_open(self, data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
        
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
        
        
        #define predictor variables
        X = np.array([frac_change, frac_high, frac_low]).transpose()
        X = sm.add_constant(X)
        
       
        def bic_general(likelihood_fn, k, X):
            """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                          of the data under the fitted model.
                   k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                            For HMM it is number of states.
                            For GMM the number of components.
                   X - array. Data that been fitted upon.
            """
            bic = np.log(len(X))*k - 2*likelihood_fn(X)
            return bic
        
        def aic_general(likelihood_fn, k, X):
            return 2*k - 2*likelihood_fn(X)

        def bic_hmmlearn(X):
            lowest_bic = np.infty
            bic = []
            aic = []
            n_states_range = range(1,10)
            #import pdb; pdb.set_trace()
            for n_components in n_states_range:
                hmm_curr = GaussianHMM(n_components=n_components, n_iter = 5,covariance_type='diag', init_params='mcs')
                hmm_curr.fit(X)

                # Calculate number of free parameters
                # free_parameters = for_means + for_covars + for_transmat + for_startprob
                # for_means & for_covars = n_features*n_components
                n_features = hmm_curr.n_features
                free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)

                bic_curr = bic_general(hmm_curr.score, free_parameters, X)
                aic_curr = aic_general(hmm_curr.score, free_parameters, X)
                aic.append(aic_curr)
                bic.append(bic_curr)
                if bic_curr < lowest_bic:
                    lowest_bic = bic_curr
                    best_hmm = hmm_curr

            return (best_hmm, bic, aic)

        best_hmm, bic, aic = bic_hmmlearn(X)
        return best_hmm, bic, aic
        
    def bic_cal_high(self, data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
        
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / high_price
        frac_high = (high_price - open_price) / high_price
        frac_low = (open_price - low_price) / high_price
        
        
        #define predictor variables
        X = np.array([frac_change, frac_high, frac_low]).transpose()
        X = sm.add_constant(X)
        
       
        def bic_general(likelihood_fn, k, X):
            """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                          of the data under the fitted model.
                   k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                            For HMM it is number of states.
                            For GMM the number of components.
                   X - array. Data that been fitted upon.
            """
            bic = np.log(len(X))*k - 2*likelihood_fn(X)
            return bic
        
        def aic_general(likelihood_fn, k, X):
            return 2*k - 2*likelihood_fn(X)

        def bic_hmmlearn(X):
            lowest_bic = np.infty
            bic = []
            aic = []
            n_states_range = range(1,10)
            #import pdb; pdb.set_trace()
            for n_components in n_states_range:
                hmm_curr = GaussianHMM(n_components=n_components,n_iter = 5,covariance_type='diag', init_params='mcs')
                hmm_curr.fit(X)

                # Calculate number of free parameters
                # free_parameters = for_means + for_covars + for_transmat + for_startprob
                # for_means & for_covars = n_features*n_components
                n_features = hmm_curr.n_features
                free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)

                bic_curr = bic_general(hmm_curr.score, free_parameters, X)
                aic_curr = aic_general(hmm_curr.score, free_parameters, X)
                aic.append(aic_curr)
                bic.append(bic_curr)
                if bic_curr < lowest_bic:
                    lowest_bic = bic_curr
                    best_hmm = hmm_curr

            return (best_hmm, bic, aic)

        best_hmm, bic, aic = bic_hmmlearn(X)
        return best_hmm, bic, aic
        
 
stock_predictor = StockPredictor(company='Nasdaq')
best_hmm, bic1, aic1 = stock_predictor.bic_cal_open(stock_predictor._train_data)
best_hmm, bic2, aic2 = stock_predictor.bic_cal_high(stock_predictor._train_data)
states = range(1,10)
plt.figure()
plt.plot(states, aic1,'ro--', label = 'AIC x')
plt.plot(states, bic1, 'bx--', label = 'BIC x')
plt.plot(states, aic2,'ro-', label = 'AIC x*')
plt.plot(states, bic2, 'bx-', label = 'BIC x*')

plt.xlabel('Number of hidden states')
plt.ylabel('Score')
plt.title('AIC and BIC scores for Exxon')
plt.legend()

#predicted_close_prices, days, actual_close_prices=stock_predictor.predict_close_prices_for_days(100, with_plot=True)