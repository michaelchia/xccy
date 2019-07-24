#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:57:34 2019

@author: m
"""
# CHANGE HERE
PRODUCT = 'EUR_4Y1Y'
MIN_SCORE = 1
# END


from xccy.modelling import Models
from xccy.vis import plot_eval

import config


models = Models.load(config.CUR_MODEL_PATH)
plot_eval(models.get_model(PRODUCT), MIN_SCORE)