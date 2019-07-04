#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:37:21 2019

@author: m
"""
import itertools
import importlib

import xccy
import xccy.data
import xccy.feature_engineering
import xccy.modelling
importlib.reload(xccy)
from xccy import data
importlib.reload(xccy.modelling)
from xccy.modelling import Models


data.initialize_data('/Users/m/Documents/xccy/data')

products = ['{}_{}'.format(ccy, prod)
            for ccy, prod in itertools.product(
                    ['AUD', 'JPY', 'EUR'], 
                    ['1Y1Y', '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '3Y2Y', '5Y5Y'])]

            
models = Models().fit(products, n_iter=500, n_jobs=-1)

models.save('models.pkl')

models2 = Models.load('models.pkl')

models2.plot_cv('AUD_1Y1Y')

import cloudpickle
import pickle
pickle.dumps(models.get_model('AUD_1Y1Y').model)
