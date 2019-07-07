#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:08:41 2019

@author: m
"""
import os

from xccy import data

os.chdir(os.path.abspath(os.path.join(__file__, os.pardir)))

DATA_DIR = 'data'
MODEL_DIR = 'models'
CUR_MODEL_NAME = 'cur_model.pkl'
CUR_MODEL_PATH = os.path.join(os.getcwd(), MODEL_DIR, CUR_MODEL_NAME)

try:
    data.refresh_data(DATA_DIR)
except Exception as e:
    print('WARNING: Data not refreshed \n'
          '{}: {}'.format(type(e).__name__, e))

data.initialize_data(DATA_DIR)