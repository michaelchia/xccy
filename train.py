#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:03:48 2019

@author: m
"""
# CHANGE HERE
CCYS = ['AUD'] # , 'JPY', 'EUR']
TERMS = ['1Y1Y'] # , '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '3Y2Y', '5Y5Y']
# END

import config

import os
import itertools
import time

from xccy.modelling import Models

products = ['{}_{}'.format(ccy, prod)
            for ccy, prod in itertools.product(CCYS, TERMS)]

try:
    models = Models.load(config.CUR_MODEL_PATH)
except:
    models = Models()

models = models.fit(products, n_iter=10, n_jobs=-1)

models.save(config.CUR_MODEL_PATH)
# archive a copy
archive_path = os.path.join(os.getcwd(), config.MODEL_DIR, 'models_{}.pkl'.format(hex(int(time.time()))[2:]))
models.save(archive_path)



