#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:57:34 2019

@author: m
"""
# CHANGE HERE
PRODUCT = 'GBP_2Y2Y'
MIN_SCORE = 2
# END


from xccy.modelling import Models

import config


models = Models.load(config.CUR_MODEL_PATH)
models.plot_cv(PRODUCT, MIN_SCORE)

for k, v in models.product_models.items():
    e = v.evaluate()
    print('{}: {:g}, {}'.format(k, e['score'], e['n']))