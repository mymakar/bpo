import gurobipy as grb
import numpy as np
from gurobipy import *

from sklearn.gaussian_process.kernels import RBF 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import Ridge
from scipy.linalg import cholesky, cho_solve, solve_triangular

class UnconstrainedRegression():
    """ Unconstraint regression """
    
    def __init__(self, model, gammau=0, gammal=0, alpha=0, kernel='linear', p=None, sig=None, standardize=True):
        """ Constructor """
        self.model = model
        self.gammau = gammau
        self.gammal = gammal
        self.alpha = alpha
        self.kernel = kernel 
        self.p = p 
        self.sig = sig 
        self.standardize = standardize



    def fit(self, x_or, y,  w=None):
        """ Fits upper and lower bounds on p(y|x) """
        
        #---preprocessing 

        if self.standardize:
            xselector = VarianceThreshold(threshold = .1).fit(x_or)
            temp_x = xselector.transform(x_or)
            xscaler = StandardScaler().fit(temp_x) 
            self.xscaler = lambda x: xscaler.transform(xselector.transform(x))
            x = self.xscaler(x_or)
        else: 
            x = x_or.copy()


        if self.kernel =='linear':
            self.kernel_fit = lambda x:x
            x = self.kernel_fit(x)

        elif self.kernel == 'poly':
            if self.p is None: 
                raise ValueError('Need polynomial value')
            
            self.kernel_fit = lambda x: np.hstack([x**i for i in range(1,self.p+1)])
            x = self.kernel_fit(x)

        elif self.kernel=='rbf':
            if self.sig is None: 
                raise ValueError('Need Length scale value')
            self.x_tr = x.copy()
            self.kernel_fit = lambda x_ts: RBF(length_scale = self.sig).__call__(x_ts, self.x_tr)
            x = self.kernel_fit(x)

        #---fitting 
        if self.model=='kr':
            reg = Ridge(alpha = self.alpha).fit(x, y)
            self.b = np.hstack([reg.coef_, reg.intercept_])
        elif self.model == 'gp':
            x[np.diag_indices_from(x)] += self.alpha
            try:
                self.L_ = cholesky(x, lower=True)
                L_inv = solve_triangular(self.L_.T,np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)

            except np.linalg.LinAlgError as exc:
                exc.args = ("The kernel is not returning a "
                            "positive definite matrix. Try gradually "
                            "increasing the 'alpha' parameter ")
                raise
            self.b = cho_solve((self.L_, True), y.copy()) 

        else: 
            raise ValueError('Model can either be gaussian process (GP) or kernel regresion (KR)')
        
        return self

    
    def predict_kr(self, x):
        """ Predicts lower and upper bounds at a point x """
        if self.standardize:
            x = self.xscaler(x)

        x = self.kernel_fit(x)

        yhat = np.dot(x, self.b[:x.shape[1]]) + self.b[-1]
        
        return yhat


    def predict_gp(self, x):
        """ Predicts lower and upper bounds at a point x """
        if self.standardize:
            x_s = self.xscaler(x)
        else: 
            x_s = x.copy()

        x = self.kernel_fit(x_s.copy())

        yhat = np.dot(x, self.b) 

        v = cho_solve((self.L_, True), x.T) 
        ycov = RBF(length_scale = self.sig).__call__(x_s.copy(), x_s.copy()) - x.dot(v)  # Line 6 
        yvar = np.diag(ycov)
        return yhat, np.sqrt(yvar)


    def predict(self, x, pred_type = "interval"):
        if self.model=='kr':
            yhat = self.predict_kr(x)
            if pred_type=="est":
                return yhat
            elif pred_type == "interval":
                yl = yhat.copy() - self.gammal
                yu = yhat.copy() + self.gammau
                return yl, yu

        else:
            yhat, sdhat = self.predict_gp(x)
            if pred_type =="est":
                return yhat
            elif pred_type == "interval":
                yl = yhat.copy() - self.gammal*sdhat.copy()
                yu = yhat.copy() + self.gammau*sdhat.copy()
                return yl, yu
            elif pred_type == "est_sd":
                return yhat, sdhat

        
    
    def set_gamma(self, gammau=None, gammal=None):
        """ Sets the gamma param (used with gamma-intervals) """
        if gammau is not None: 
            self.gammau = gammau
        if gammal is not None: 
            self.gammal = gammal



    def fit_conformal(self, q, x, y, w=None):
        ''' Only used with conformal intervals'''

        if self.standardize:
            x = self.xscaler(x)

        x = self.kernel_fit(x)

        yhat = np.dot(x, self.b[:x.shape[1]]) + self.b[-1]

        resid = np.abs((y.ravel()-yhat.ravel()))
        assert len(resid.shape) ==1
        resid = np.sort(resid)
        qid = np.floor((resid.shape[0] + 1)*(1-q))
        try: 
            self.gammau = self.gammal = resid[int(qid)]
        except:
            self.gammau = self.gammal = resid[int(qid)-1]




