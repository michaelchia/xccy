#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:03:48 2019

@author: m
"""
# CHANGE HERE
CCYS = ['AUD', 'JPY', 'EUR', 'NZD', 'GBP']
TERMS = ['1Y1Y', '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '5Y5Y']
ITERATIONS = 500
# END

import config

import os
import itertools
import time

import numpy as np
import dateutil

from xccy.modelling import Models
from xccy.data import global_data

products = ['{}_{}'.format(ccy, prod)
            for ccy, prod in itertools.product(CCYS, TERMS)]

try:
    models = Models.load(config.CUR_MODEL_PATH)
except:
    models = Models()

# latest_date minus one year
date_split = np.max(global_data.get_time_series()) - dateutil.relativedelta.relativedelta(years=1)

models = models.fit(products, date_split=date_split, n_iter=ITERATIONS, n_jobs=-1)

try:
    os.mkdir(config.MODEL_DIR)
except FileExistsError:
    pass

models.save(config.CUR_MODEL_PATH)
# archive a copy
archive_path = os.path.join(os.getcwd(), config.MODEL_DIR, 'models_{}.pkl'.format(hex(int(time.time()))[2:]))
models.save(archive_path)