#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:57:34 2019

@author: m
"""
# CHANGE HERE
PRODUCT = 'AUD_1Y1Y'
# END


from xccy import data
from xccy.modelling import Models

data.initialize_data('data')
models = Models.load('models.pkl')
models.plot_cv(PRODUCT)