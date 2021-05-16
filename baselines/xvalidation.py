import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from copy import deepcopy
from .regression import *

def fcr(y,l,u,w=None):
	if w is None:
		return 1-np.mean((y>l)*(y<u))
	else:
		temp_fcr = np.logical_or((y<l), (y>u))
		return np.sum(w*temp_fcr)

def fcr_avg(y,l,u):
	fcr0 = 1-np.mean((y[0]>l[0])*(y[0]<u[0]))
	fcr1 = 1-np.mean((y[1]>l[1])*(y[1]<u[1]))
	return (fcr0 + fcr1)/2

def iw(l,u, w=None, op='mean'):
	if op=='mean':
		if w is None:
			return np.mean((u-l)**2)
		else:
			return np.sum(w*((u-l)**2))
	else:
		return np.max(u-l)

def iw_avg(l,u, op='mean'):
	if op=='mean':
		return np.mean(np.abs(u[0]-l[0]) + np.abs(u[1]-l[1]) )
	else:
		return np.max(np.abs(u[0]-l[0]) + np.abs(u[1]-l[1]) )

def get_mse(y, yhat, w):
	if w is None:
		return np.mean((y-yhat)**2)
	else:
		return np.sum(w*(y-yhat)**2)



def reg_sl_xv(model, nfolds, data, params, tried_params=None, oracle = False):

	#-----get all the parameters to cross validate over
	n_params = len(params[0])
	all_combos = np.array(np.meshgrid(*params[1])).T.reshape(-1,n_params)
	xv_df = pd.DataFrame(data=np.hstack([all_combos, np.full((all_combos.shape[0], nfolds), np.nan)]), \
		index = np.array(range(all_combos.shape[0])), \
		columns = [key for key in params[0]] + [f'mse_{i}' for i in range(nfolds)] )

	#-----start xvalidation
	if oracle:
		x, y, w, xo, Y0o, Y1o = data
	else:
		x, y, w = data

	if 'p' in params[0]:
		kern_param = 'p'
		kern_str = 'poly'
	elif 'sig' in params[0]:
		kern_param = 'sig'
		kern_str = 'rbf'
	else:
		kern_param = 'kern_param'
		kern_str = 'linear'


	kf = KFold(n_splits=nfolds, random_state = 0)

	masked_params = [True]*all_combos.shape[0]


	for combi in range(all_combos.shape[0]):
		if ((tried_params is not None) and ((tried_params['alpha'] == xv_df.loc[combi,'alpha']) \
			& (tried_params[kern_param] == xv_df.loc[combi,kern_param]) \
			& (tried_params['gamma'] == xv_df.loc[combi,'gamma'])).any()):
				masked_params[combi] = False
				continue

		for kfi, spliti in enumerate(kf.split(x)):
			train_index, test_index = spliti[0], spliti[1]
			x_tr, x_ts = x[train_index], x[test_index]

			y_tr, y_ts = y[train_index].ravel(), y[test_index].ravel()
			if w is None:
				w_tr, w_ts = None, None
			else:
				w_tr, w_ts = w[train_index], w[test_index]

				w_tr = np.mean(w_tr)/w_tr
				w_tr = w_tr/np.sum(w_tr)

				w_ts = np.mean(w_ts)/w_ts
				w_ts = w_ts/np.sum(w_ts)


			if model =="kr":
				if 'p' in params[0]:
					M = UnconstrainedRegression(model= 'kr', gammau = 0, gammal = 0, alpha = xv_df.loc[combi,'alpha'], kernel='poly', p=int(xv_df.loc[combi,'p']),  standardize=True)
				elif 'sig' in params[0]:
					M = UnconstrainedRegression(model= 'kr', gammau = 0, gammal = 0, alpha = xv_df.loc[combi,'alpha'], kernel='rbf', sig=xv_df.loc[combi,'sig'],  standardize=True)
				else:
					M = UnconstrainedRegression(model= 'kr', gammau = 0, gammal = 0, alpha = xv_df.loc[combi,'alpha'], kernel='linear', standardize=True)


			elif model == "gp":
				if 'p' in params[0]:
					print(xv_df.loc[combi,'alpha'])
					M = UnconstrainedRegression(model= 'gp', gammau = 0, gammal = 0, alpha = xv_df.loc[combi,'alpha'], kernel='poly', p=int(xv_df.loc[combi,'p']),  standardize=True)
				else:
					M = UnconstrainedRegression(model= 'gp', gammau = 0, gammal = 0, alpha = xv_df.loc[combi,'alpha'], kernel='rbf', sig=xv_df.loc[combi,'sig'],  standardize=True)


			M.fit(x_tr,y_tr,w_tr)

			mse_v = get_mse(y_ts, M.predict(x_ts, pred_type = "est"), w_ts)

			if oracle:
				print("not implemented")

			xv_df.loc[combi, f'mse_{kfi}'] = mse_v

			temp_mse =  np.mean(xv_df.loc[combi, [f'mse_{i}' for i in range(nfolds)]])

	xv_df = xv_df[masked_params]
	xv_df = xv_df.append(tried_params, ignore_index = True)
	xv_df['mse'] =  np.mean(xv_df[[f'mse_{i}' for i in range(nfolds)]], axis = 1)
	best_erm = xv_df[(xv_df.mse == np.min(xv_df.mse))]
	if best_erm.shape[0]>1:
		best_erm = best_erm.iloc[0]

	return xv_df, best_erm


def gint_sl_xv(gammaparams, data, yhat, gt_data= None):
	gamprod = pd.core.reshape.util.cartesian_product([val for val in gammaparams.values()])
	xv_gamma = pd.DataFrame({key:gamprod[ki] for ki, key in enumerate(gammaparams.keys())})
	if 'gamma' in xv_gamma.columns:
		xv_gamma['gammal'] = xv_gamma['gammau'] = xv_gamma['gamma']
	xv_gamma['fcr'] = np.nan
	xv_gamma['meaniw'] = np.nan

	yhat_or  = yhat.copy()

	y_ts, w_ts = data
	if w_ts is not None:
		w_ts = np.mean(w_ts)/w_ts
		w_ts = w_ts/np.sum(w_ts)
		w_ts = w_ts.ravel()

	for gi in range(xv_gamma.shape[0]):
		lb, ub = yhat.copy() - xv_gamma.gammal.iloc[gi], yhat.copy() + xv_gamma.gammau.iloc[gi]
		xv_gamma.loc[gi, f'fcr'] = fcr(y_ts.ravel(), lb.ravel(), ub.ravel(), w_ts)
		xv_gamma.loc[gi, f'meaniw']  = iw(lb.ravel(), ub.ravel(), w_ts)

		if gt_data is not None:
			y_gt, yh_fcr= gt_data
			lb, ub = y_gt.copy() - xv_gamma.gammal.iloc[gi], y_gt.copy() + xv_gamma.gammau.iloc[gi]
			xv_gamma.loc[gi, f'fcr'] = fcr(y_gt.ravel(), lb.ravel(), ub.ravel(), None)

	return xv_gamma

def calci_sl_xv(gammavals, data, yhat, ystd, gt_data = None):
	xv_gamma = pd.DataFrame({'gamma': gammavals})
	xv_gamma['fcr'] = np.nan
	xv_gamma['meaniw'] = np.nan

	yhat_or  = yhat.copy()

	y_ts, w_ts = data
	if w_ts is None:
		w_ts = np.ones(y_ts.shape[0])/y_ts.shape[0]
	else:
		w_ts = np.mean(w_ts)/w_ts
		w_ts = w_ts/np.sum(w_ts)

	for gi in range(xv_gamma.shape[0]):
		lb = yhat.copy() - xv_gamma.gamma.iloc[gi]*ystd.copy()
		ub = yhat.copy() + xv_gamma.gamma.iloc[gi]*ystd.copy()
		xv_gamma.loc[gi, f'meaniw']  = iw(lb.ravel(), ub.ravel(), w_ts.ravel())
		xv_gamma.loc[gi, f'fcr'] = fcr(y_ts.ravel(), lb.ravel(), ub.ravel(), w_ts.ravel())

		if gt_data is not None:
			y_gt, yh_fcr, ys_fcr = gt_data
			lb = yh_fcr.copy() - xv_gamma.gamma.iloc[gi]*ys_fcr.copy()
			ub = yh_fcr.copy() + xv_gamma.gamma.iloc[gi]*ys_fcr.copy()
			xv_gamma.loc[gi, f'fcr'] = fcr(y_gt.ravel(), lb.ravel(), ub.ravel(), None)


	return xv_gamma

