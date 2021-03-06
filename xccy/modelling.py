#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:18:33 2019

@author: m
"""
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy
from scipy.stats.distributions import randint, uniform

from .data import ProductData, Product, PAY, RECEIVE
from .feature_engineering import FeatSelector, FeatEng, LabelEng, FastLabelEng


MIN_DATA_DATE = datetime.datetime(2015,1,1)
SPLIT_DATA = datetime.datetime(2018,6,1)
N_ITER = 500

LOOKAHEAD = 15
WINDOW = 12
COST = 1
LOSS_PENALTY = 0.25 

                            
SCORE_THRESHOLD = 2
TRADE_WAIT = 7
SCORER_MIN_TRADE = 10

CCY_DIRECTION_MAP = {
    'AUD': RECEIVE,
    'EUR': RECEIVE,
    'JPY': PAY,
    'GBP': RECEIVE,
    'NZD': RECEIVE,
}

FEAT_ENG = FeatEng
LABEL_ENG = FastLabelEng  # LabelEng, FastLabelEng


class Models:
    def __init__(self):
        self.product_models = {}
    
    def fit(self, products, date_split=SPLIT_DATA, n_iter=N_ITER, n_jobs=-1):
        products = [Product.from_string(p) if isinstance(p, str) else p
                    for p in products]
        models = {p.to_string(ccy=True): ProductModel(p).fit(date_split, n_iter, n_jobs)
                  for p in products}
        self.product_models.update(models)
        return self
    
    def predict_latest(self):
        from xccy.data import global_data
        dates = [np.max(global_data.get_time_series())]
        pred = {k: v[0] for k, v in self.predict(dates).items()}
        return dates[0], pred
        
    def predict(self, dates=None, products=None):
        return {k: v.predict(dates)
                for k, v in self.product_models.items()}
    
    def get_model(self, product):
        if isinstance(product, str):
            product = Product.from_string(product)
        return self.product_models[product.to_string(ccy=True)]
        
    def plot(self, product, min_score=1):
        pass
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
      
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    
class ProductModel:
    def __init__(self, product, 
                 lookahead=LOOKAHEAD, 
                 window=WINDOW,
                 cost=COST,
                 loss_penalty=LOSS_PENALTY):
        self.product = product
        self.model = Classifier()
        self.fe = FEAT_ENG()
        self.labler = LABEL_ENG(lookahead=lookahead, 
                                window=window,
                                cost=cost,
                                loss_penalty=loss_penalty,
                                side=CCY_DIRECTION_MAP[product.ccy])
        
    def fit(self, date_split, n_iter=N_ITER, n_jobs=-1):
        self.date_split_ = date_split
        print('Fitting {}'.format(self.product.to_string()))
        tdata = self._training_data(date_split)
        self.model.fit(**tdata, n_iter=n_iter, n_jobs=n_jobs)
        # print evaluation
        cv_data = self.model.cv_data_
        score = Scorer().evaluate(cv_data['y'], 
                                  cv_data['y_pred'], 
                                  min_trades=10)['score']
        print('  Eval score: {:2g}'.format(score))
        return self
    
    def predict(self, dates=None):
        pdata = ProductData(self.product, dates=dates)
        features = self.fe.get_features(pdata)
        return self.model.predict(features)
    
    @property
    def cv_trades(self):
        cv_data = self.model.cv_data_
        return Scorer().trades(cv_data['y'], cv_data['y_pred'], min_trades=10)        

    def _training_data(self, date_split):
        pdata = ProductData(self.product, min_date=MIN_DATA_DATE)
        return make_training_data(
                    pdata,
                    self.fe,
                    self.labler,
                    date_split=date_split,
                )
        

def make_training_data(product_data, feat_eng, label_eng, date_split):
    features = feat_eng.get_features(product_data)
    labels = label_eng.get_labels(product_data)
    
    Xy = features.join(labels.to_frame()).infer_objects()
    Xy = Xy.dropna()
    X = Xy[features.columns]
    y = Xy[labels.name]
    
    def make_cv_split(X, date_split):
        ls = list(X.index < date_split)
        return [(np.array([i for i, x in enumerate(ls) if x]),
                 np.array([i for i, x in enumerate(ls) if not x]))]
    cv_split = make_cv_split(X, date_split)
    return dict(
        X=X,
        y=y,
        cv_split=cv_split
    )


class SubModel:
    def fit(self, X, y, cv_split, n_iter=10, n_jobs=None):
        cv_search = self.hyperparam_search(cv_split, n_iter, n_jobs=n_jobs)
        cv_search.fit(X, y) #, est__sample_weight=sample_weight)
        # self.search_ = cv_search
        self.model_ = self.pipeline.set_params(**cv_search.best_params_)
        train_idx = list(cv_split)[0][0]
        X_train = X.iloc[train_idx,]
        y_train = y[train_idx]
        self.model_.fit(X_train, y_train)
        self.features_ = self.model_.named_steps['fs'].features_
        cv_idx = list(cv_split)[0][1]
        X_cv = X.iloc[cv_idx,]
        y_cv = y[cv_idx]
        pred_cv = self.model_predict(X_cv)
        self.linear_model_ = self.make_linear_model()
        self.linear_model_.fit(pred_cv, y_cv)
        pred_raw = pd.Series(pred_cv[:,0], index=y_cv.index)
        pred = pd.Series(list(self.linear_predict(pred_cv)), index=y_cv.index)
        self.cv_data_ = pd.DataFrame({'y': y_cv, 'y_pred': pred, 'y_score': pred_raw})
        return self
        
    def predict(self, X):
        X = X.dropna()
        pred = self.model_predict(X)
        pred = self.linear_predict(pred)
        return pd.Series(pred, index=X.index)
    
    def predict_raw(self, X):
        X = X.dropna()
        pred = self.model_predict(X)[:,0]
        return pd.Series(pred, index=X.index)
    
    @property
    def param_distributions(self):
        param_space = self.est_param_distributions
        param_space.update({k: v for k, v in FeatSelector.param_space().items()})
        return param_space
    
    @property
    def pipeline(self):
        return Pipeline(memory=None,
                        steps=[('fs', self.feat_selector),
                               #('scaler', MinMaxScaler()),
                               ('est', self.estimator)])
    
    @property
    def feat_selector(self):
        return FeatSelector()
    
    def hyperparam_search(self, cv, n_iter, n_jobs=None):
        # BayesSearchCV
        return RandomizedSearchCV(self.pipeline, 
                                  self.param_distributions, 
                                  n_iter=n_iter, 
                                  scoring=self.scoring, 
                                  fit_params=None, 
                                  n_jobs=n_jobs, 
                                  refit=False, 
                                  cv=cv, 
                                  verbose=0,
                                  random_state=1)
    

class Classifier(SubModel):
    DEFAULT_THRESHOLD = 0.5
    
    def model_predict(self, X):
        return self.model_.predict_proba(X)[:,1].reshape(-1, 1)
    
    def linear_predict(self, X):
        return self.linear_model_.predict(X)

    def make_linear_model(self):
        return IsoPchipRegression()
    
    @property    
    def scoring(self):
        return Scorer().scorer # 'roc_auc'
    
    @property
    def estimator(self):
        return GBC()
    
    @property
    def est_param_distributions(self):
        return {
                'est__n_estimators': randint(20, 200),
                'est__min_change': uniform(0,4),
                'est__use_weight': [True, False],
            }


class GBC(GradientBoostingClassifier):
    def __init__(self, n_estimators=10, min_change=0, use_weight=False, min_weight=1):
        self.n_estimators = n_estimators
        self.min_change = min_change
        self.use_weight = use_weight
        self.min_weight = min_weight
        super().__init__(n_estimators=n_estimators)
    
    def fit(self, X, y):
        thres = min(self.min_change, y.max())
        y = y.map(lambda x: int(x >= thres))
        if self.use_weight:
            sample_weight = y.map(lambda x: max(abs(x - thres), self.min_weight)/y.max())
            sample_weight = sample_weight * len(sample_weight) / np.sum(sample_weight)
        else:
            sample_weight = None
        return super().fit(X, y, sample_weight=sample_weight)


class IsoPchipRegression:        
    def fit(self, X, y):
        x = X[:,0]
        ir = IsotonicRegression()
        ir.fit(x, y)
        xi, yi = self._ir_roots(ir)
        self.pchip_ = scipy.interpolate.PchipInterpolator(xi, yi) \
                      if len(xi) > 1 else None
        return self
        
    def predict(self, X):
        x = X[:,0]
        if self.pchip_:
            return self.pchip_(x)
        return x
        
    def _ir_roots(self, ir):
        prev_y = None
        prev_x = None
        xs = []
        ys = []
        try:
            for x, y in zip(ir.f_.x, ir.f_.y):
                if y == prev_y:
                    xs.append((x + prev_x) / 2)
                    ys.append(y)
                prev_x, prev_y = x, y
            return np.array(xs), np.array(ys)
        except AttributeError:
            return np.array([]), np.array([])



class Scorer:
    def __init__(self, threshold=SCORE_THRESHOLD, min_days=TRADE_WAIT):
        self.threshold = threshold
        self.min_days = min_days
        
    def trades(self, y, pred_y, min_trades=0, return_thres=False):
        INC = 0.1
        def trades(min_bp):
            pl = y[pred_y > min_bp].sort_index()
            mask = []
            prev_date = datetime.datetime(2000,1,1)
            for date in pl.index:
                b = (date - prev_date).days > self.min_days
                mask.append(b)
                if b:
                    prev_date = date
            return pl[mask]
        thres = self.threshold
        t = trades(thres)
        while len(t) < min_trades and thres > 0:
            t = trades(thres)
            thres -= INC
        if return_thres:
            return t, thres + INC
        return t
    
    def evaluate(self, y, pred_y, min_trades=1):
        trades, thres = self.trades(y, pred_y, min_trades=min_trades, return_thres=True)
        return dict(
            score=np.sum(trades),
            n=len(trades),
            threshold=thres,
            trades=dict(
                mean=np.mean(trades),
                pos=np.mean(trades>0),
                mean_pos=np.mean(trades[trades>0]),
                neg=np.mean(trades<=0),
                mean_neg=np.mean(trades[trades<=0])
            ),
            base=dict(
                mean=np.mean(y),
                pos=np.mean(y>0),
                mean_pos=np.mean(y[y>0]),
                neg=np.mean(y<=0),
                mean_neg=np.mean(y[y<=0])                
            )
        )
    
    def score(self, y, pred_y):
        return np.sum(self.trades(y, pred_y))
    
    @property
    def scorer(self):
        def score(y, y_prob):
            y_prob = y_prob[:,1].reshape(-1, 1)
            pred_y = IsoPchipRegression().fit(y_prob, y).predict(y_prob)
            return np.sum(self.trades(y, pred_y, min_trades=SCORER_MIN_TRADE))
        return make_scorer(score, needs_proba=True)