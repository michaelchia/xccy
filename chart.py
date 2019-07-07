#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:57:34 2019

@author: m
"""
# CHANGE HERE
PRODUCT = 'AUD_1Y1Y'
MIN_SCORE = 1
# END


from xccy.modelling import Models

import config


models = Models.load(Cconfig.UR_MODEL_PATH)
models.plot_cv(PRODUCT, MIN_SCORE)