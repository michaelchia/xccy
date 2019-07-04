#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:27:56 2019

@author: m
"""
from xccy import data
from xccy.modelling import Models

data.initialize_data('data')
models = Models.load('models.pkl')

dt, pred = models.predict_latest()
sorted_pred = sorted(pred.items(), key=lambda x: -x[1])
print('Date: {}'.format(dt.strftime('%d-%m-%Y')))
for prod, score in sorted_pred:
    print('{}: {:2g}'.format(prod, score))