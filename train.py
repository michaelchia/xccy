#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:03:48 2019

@author: m
"""
import itertools
import time

from xccy import data
from xccy.modelling import Models

# CHANGE HERE
CCYS = ['AUD', 'JPY', 'EUR']
TERMS = ['1Y1Y', '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '3Y2Y', '5Y5Y']
# END

data.initialize_data('data')

products = ['{}_{}'.format(ccy, prod)
            for ccy, prod in itertools.product(CCYS, TERMS)]

try:
    models = Models.load('models.pkl')
except:
    models = Models()

models = models.fit(products, n_iter=500, n_jobs=-1)

models.save('models.pkl')
models.save('archive/models_{}.pkl'.format(hex(int(time.time()))[2:]))