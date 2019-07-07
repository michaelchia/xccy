#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:27:56 2019

@author: m
"""
import config

from xccy.modelling import Models



models = Models.load(config.CUR_MODEL_PATH)

dt, pred = models.predict_latest()
sorted_pred = sorted(pred.items(), key=lambda x: -x[1])
print('Date: {}'.format(dt.strftime('%d-%m-%Y')))
for prod, score in sorted_pred:
    print('{}: {:2g}'.format(prod, score))