import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
import multiprocessing as mp
from functools import partial
from tqdm import *
from .bound import *


import matplotlib.pyplot as plt

def fcr(y,l,u,w=None):
	temp_fcr = np.logical_or((y<l), (y>u))
	if w is None:
		return np.mean(temp_fcr)
	return np.sum(w*temp_fcr)

def iw(l,u, w=None, op='mean'):
	if op=='mean':
		if w is None:
			return np.mean((u-l)**2)
		else:
			return np.sum(w*((u-l)**2))
	else:
		return np.max(u-l)

def d_sl_xv_helper(config, data, data_ts_gt, splits, all_gammas, loss, agg):

	# empty df
	cnfg_df = pd.DataFrame({col: [] for col in [key for key in config.keys()] \
		+ [col for col in all_gammas.columns] + [f'fcr_{i}' for i in range(len(splits))] \
		+ [f'meaniw_{i}' for i in range(len(splits))] \
		+ [f'maxiw_{i}' for i in range(len(splits))]})

	# populate with params
	cnfg_df[[col for col in all_gammas.columns]] = all_gammas
	for col, val in  config.items():
		cnfg_df[col] = val

	if 'lamdau' not in cnfg_df.columns:
		cnfg_df.rename(columns={'lamda': 'lamdau'}, inplace = True)
		cnfg_df['lamdal'] = cnfg_df['lamdau']

	if 'alphau' not in cnfg_df.columns:
		cnfg_df.rename(columns={'alpha': 'alphau'}, inplace = True)
		cnfg_df['alphal'] = cnfg_df['alphau']

	if 'gammau' not in cnfg_df.columns:
		cnfg_df.rename(columns={'gamma': 'gammau'}, inplace = True)
		cnfg_df['gammal'] = cnfg_df['gammau']

	# update config
	config = cnfg_df.iloc[0].copy()
	config['gammau'] = 0
	config['gammal'] = 0

	config.drop([f'fcr_{i}' for i in range(len(splits))],  inplace = True)
	config.drop([f'meaniw_{i}' for i in range(len(splits))],  inplace = True)
	config.drop([f'maxiw_{i}' for i in range(len(splits))],  inplace = True)

	config = config.to_dict()
	x, y, w = data

	for si, split in enumerate(splits):
		train_index, test_index = split[0], split[1]
		x_tr, x_ts = x[train_index], x[test_index]

		y_tr, y_ts = y[train_index].ravel(), y[test_index].ravel()

		if w is None:
			w_tr, w_ts = None, None
		else:
			w_tr, w_ts = w[train_index].copy(), w[test_index].copy()

			w_tr = np.mean(w_tr)/w_tr
			w_tr = w_tr/np.sum(w_tr)

			w_ts = np.mean(w_ts)/w_ts
			w_ts = w_ts/np.sum(w_ts)

		try:
			M = LinearBoundRegression(loss = loss, agg= agg, **config, standardize = True)
			M.fit(x_tr,y_tr,w_tr)
		except:
			print("Gurobi error. Model was infesable with specified params, try larger alpha")
			continue

		for gid in range(cnfg_df.shape[0]):
			curr_cnfg = cnfg_df.iloc[gid].copy()
			curr_cnfg = curr_cnfg[['gammau',  'gammal']]
			curr_cnfg = curr_cnfg.to_dict()
			M.set_params(**curr_cnfg)
			if data_ts_gt is not None:
				x_ts_gt, y_ts_gt = data_ts_gt
				fcr_v = fcr(y_ts_gt, M.predict(x_ts_gt)[0], M.predict(x_ts_gt)[1], None)
			else:
				fcr_v = fcr(y_ts, M.predict(x_ts)[0], M.predict(x_ts)[1], w_ts)
			iw_v = iw(M.predict(x_ts)[0], M.predict(x_ts)[1], w_ts)
			iwm_v = iw(M.predict(x_ts)[0], M.predict(x_ts)[1], op='max')


			cnfg_df.loc[gid, f'fcr_{si}'] = fcr_v
			cnfg_df.loc[gid, f'meaniw_{si}'] = iw_v
			cnfg_df.loc[gid, f'maxiw_{si}'] = iwm_v



	cnfg_df['fcr'] = np.nanmean(cnfg_df[[f'fcr_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['meaniw'] = np.nanmean(cnfg_df[[f'meaniw_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['maxiw'] = np.nanmean(cnfg_df[[f'maxiw_{i}' for i in range(len(splits))]], axis = 1)

	cnfg_df.drop([f'fcr_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'meaniw_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'maxiw_{i}' for i in range(len(splits))], axis = 1, inplace = True)

	return cnfg_df





def c_xv_helper(config, data, data_ts_gt, splits, all_gammas, loss, agg):
	# populate with params
	# empty df
	perf_cols = [f'fcr_{i}' for i in range(len(splits))] \
		+ [f'fcr1_{i}' for i in range(len(splits))] \
		+ [f'fcr0_{i}' for i in range(len(splits))] \
		+ [f'meaniw_{i}' for i in range(len(splits))] \
		+ [f'meaniw1_{i}' for i in range(len(splits))] \
		+ [f'meaniw0_{i}' for i in range(len(splits))] \
		+ [f'maxiw_{i}' for i in range(len(splits))]


	cnfg_df = pd.DataFrame({col: [] for col in [key for key in config.keys()] \
		+ [col for col in all_gammas.columns] + perf_cols})



	cnfg_df[[col for col in all_gammas.columns]] = all_gammas

	for col, val in  config.items():
		cnfg_df[col] = val


	if 'lamdau1' not in cnfg_df.columns:
		if 'lamda1' in cnfg_df.columns:
			cnfg_df.rename(columns={'lamda1': 'lamdau1'}, inplace = True)
			cnfg_df['lamdal1'] = cnfg_df['lamdau1']
		else:
			cnfg_df.rename(columns={'lamda': 'lamdau1'}, inplace = True)
			cnfg_df['lamdal1'] = cnfg_df['lamdau1']

	if 'lamdau0' not in cnfg_df.columns:
		if 'lamda0' in cnfg_df.columns:
			cnfg_df.rename(columns={'lamda0': 'lamdau0'}, inplace = True)
			cnfg_df['lamdal0'] = cnfg_df['lamdau0']
		else:
			cnfg_df['lamdau0'] = cnfg_df['lamdal0'] = cnfg_df['lamdau1']



	if 'alphau1' not in cnfg_df.columns:
		if 'alpha1' in cnfg_df.columns:
			cnfg_df.rename(columns={'alpha1': 'alphau1'}, inplace = True)
			cnfg_df['alphal1'] = cnfg_df['alphau1']
		else:
			cnfg_df.rename(columns={'alpha': 'alphau1'}, inplace = True)
			cnfg_df['alphal1'] = cnfg_df['alphau1']

	if 'alphau0' not in cnfg_df.columns:
		if 'alpha0' in cnfg_df.columns:
			cnfg_df.rename(columns={'alpha0': 'alphau0'}, inplace = True)
			cnfg_df['alphal0'] = cnfg_df['alphau0']
		else:
			cnfg_df['alphau0'] = cnfg_df['alphal0'] = cnfg_df['alphau1']


	if 'gammau1' not in cnfg_df.columns:
		if 'gamma1' in cnfg_df.columns:
			cnfg_df.rename(columns={'gamma1': 'gammau1'}, inplace = True)
			cnfg_df['gammal1'] = cnfg_df['gammau1']
		else:
			cnfg_df.rename(columns={'gamma': 'gammau1'}, inplace = True)
			cnfg_df['gammal1'] = cnfg_df['gammau1']

	if 'gammau0' not in cnfg_df.columns:
		if 'gamma0' in cnfg_df.columns:
			cnfg_df.rename(columns={'gamma0': 'gammau0'}, inplace = True)
			cnfg_df['gammal0'] = cnfg_df['gammau0']
		else:
			cnfg_df['gammau0'] = cnfg_df['gammal0'] = cnfg_df['gammau1']



	# update config
	config = cnfg_df.iloc[0].copy()
	config['gammau0'] = 0
	config['gammal0'] = 0

	config['gammau1'] = 0
	config['gammal1'] = 0

	config.drop(perf_cols,  inplace = True)
	config = config.to_dict()


	x, y, t, w = data
	for si, split in enumerate(splits):
		train_index, test_index = split[0], split[1]
		x_tr, x_ts = x[train_index], x[test_index]
		y_tr, y_ts = y[train_index].ravel(), y[test_index].ravel()
		t_tr, t_ts = t[train_index].ravel(), t[test_index].ravel()

		if w is None:
			w_tr, w_ts = None, None
			w_tr_all = None
			w_ts_all = None

		else:
			w_tr, w_ts = w[train_index], w[test_index]
			w0_tr, w1_tr = w_tr[t_tr.ravel()==0], w_tr[t_tr.ravel()==1]
			w0_tr = 1-w0_tr
			w0_tr = np.mean(w0_tr)/w0_tr
			w0_tr = w0_tr/np.sum(w0_tr)

			w1_tr = np.mean(w1_tr)/w1_tr
			w1_tr = w1_tr/np.sum(w1_tr)

			w_tr_all = [w0_tr,w1_tr]


			w_ts_all = t_ts.ravel()*w_ts + (1-t_ts.ravel())*(1-w_ts)
			w_ts_all[t_ts.ravel()==1] = np.mean(w_ts_all[t_ts.ravel()==1])/w_ts_all[t_ts.ravel()==1]
			w_ts_all[t_ts.ravel()==1] = w_ts_all[t_ts.ravel()==1]/np.sum(w_ts_all[t_ts.ravel()==1])


			w_ts_all[t_ts.ravel()==0] = np.mean(w_ts_all[t_ts.ravel()==0])/w_ts_all[t_ts.ravel()==0]
			w_ts_all[t_ts.ravel()==0] = w_ts_all[t_ts.ravel()==0]/np.sum(w_ts_all[t_ts.ravel()==0])



		x0_tr, x1_tr = x_tr[t_tr.ravel()==0], x_tr[t_tr.ravel()==1]
		y0_tr, y1_tr = y_tr[t_tr.ravel()==0], y_tr[t_tr.ravel()==1]

		try:

			M = LinearBoundRegressionCombined(loss = loss, agg= agg, **config, standardize = True)
			M.fit([x0_tr, x1_tr], [y0_tr, y1_tr], w_tr_all)
		except:
			print("Gurobi error. Model was infesable with specified params, try larger alpha")
			continue

		for gid in range(cnfg_df.shape[0]):
			curr_cnfg = cnfg_df.iloc[gid].copy()
			curr_cnfg = curr_cnfg[['gammau1', 'gammau0', 'gammal1', 'gammal0']]
			curr_cnfg = curr_cnfg.to_dict()

			M.set_params(**curr_cnfg)

			l0, u0, l1, u1 = M.predict(x_ts)
			lf = t_ts*l1 + (1-t_ts)*l0
			uf = t_ts*u1 + (1-t_ts)*u0

			if data_ts_gt is not None:
				x_ts_gt, Y0_ts_gt, Y1_ts_gt = data_ts_gt
				l0_ts_gt, u0_ts_gt, l1_ts_gt, u1_ts_gt = M.predict(x_ts_gt)
				fcr_v = .5*fcr(Y0_ts_gt, l0_ts_gt, u0_ts_gt) + .5*fcr(Y1_ts_gt, l1_ts_gt, u1_ts_gt)
			else:
				fcr_v = fcr(y_ts, lf, uf, w_ts_all)/2
			iw_v = (iw(l1, u1, w_ts_all) + iw(l0, u0, w_ts_all))/2
			# iw_v = iw(y_ts, lf, uf, w_ts_all)
			iwm_v = (iw(l1, u1, None, op="max") + iw(l0, u0, None, op = "max"))/2

			if w_ts_all is None:
				fcr1_v = fcr(y_ts[t_ts.ravel()==1], lf[t_ts.ravel()==1], uf[t_ts.ravel()==1], None)
				fcr0_v = fcr(y_ts[t_ts.ravel()==0], lf[t_ts.ravel()==0], uf[t_ts.ravel()==0], None)
				iw1_v = iw(lf[t_ts.ravel()==1], uf[t_ts.ravel()==1], None)
				iw0_v = iw(lf[t_ts.ravel()==0], uf[t_ts.ravel()==0], None)
			else:
				fcr1_v = fcr(y_ts[t_ts.ravel()==1], lf[t_ts.ravel()==1], uf[t_ts.ravel()==1], w_ts_all[t_ts.ravel()==1])
				fcr0_v = fcr(y_ts[t_ts.ravel()==0], lf[t_ts.ravel()==0], uf[t_ts.ravel()==0], w_ts_all[t_ts.ravel()==0])
				iw1_v = iw( lf[t_ts.ravel()==1], uf[t_ts.ravel()==1], w_ts_all[t_ts.ravel()==1])
				iw0_v = iw( lf[t_ts.ravel()==0], uf[t_ts.ravel()==0], w_ts_all[t_ts.ravel()==0])

			cnfg_df.loc[gid, f'fcr_{si}'] = fcr_v
			cnfg_df.loc[gid, f'fcr1_{si}'] = fcr1_v
			cnfg_df.loc[gid, f'fcr0_{si}'] = fcr0_v
			cnfg_df.loc[gid, f'meaniw_{si}'] = iw_v
			cnfg_df.loc[gid, f'meaniw1_{si}'] = iw1_v
			cnfg_df.loc[gid, f'meaniw0_{si}'] = iw0_v
			cnfg_df.loc[gid, f'maxiw_{si}'] = iwm_v


	cnfg_df['fcr'] = np.nanmean(cnfg_df[[f'fcr_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['fcr1'] = np.nanmean(cnfg_df[[f'fcr1_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['fcr0'] = np.nanmean(cnfg_df[[f'fcr0_{i}' for i in range(len(splits))]], axis = 1)


	cnfg_df['meaniw'] = np.nanmean(cnfg_df[[f'meaniw_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['meaniw1'] = np.nanmean(cnfg_df[[f'meaniw1_{i}' for i in range(len(splits))]], axis = 1)
	cnfg_df['meaniw0'] = np.nanmean(cnfg_df[[f'meaniw0_{i}' for i in range(len(splits))]], axis = 1)

	cnfg_df['maxiw'] = np.nanmean(cnfg_df[[f'maxiw_{i}' for i in range(len(splits))]], axis = 1)

	cnfg_df.drop([f'fcr_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'fcr1_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'fcr0_{i}' for i in range(len(splits))], axis = 1, inplace = True)

	cnfg_df.drop([f'meaniw_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'meaniw1_{i}' for i in range(len(splits))], axis = 1, inplace = True)
	cnfg_df.drop([f'meaniw0_{i}' for i in range(len(splits))], axis = 1, inplace = True)

	cnfg_df.drop([f'maxiw_{i}' for i in range(len(splits))], axis = 1, inplace = True)

	return cnfg_df





def bp_xv(version, nfolds, data, params, gammaparams, data_ts_gt = None, loss = 'square', agg='mean', njobs = 0, tried_params=None, verbose = False, temp_file = None, nrand=0):
	# generate all combinations of params
	all_configs = list(ParameterGrid(params))
	gamprod = pd.core.reshape.util.cartesian_product([val for val in gammaparams.values()])
	all_gammas = pd.DataFrame({key:gamprod[ki] for ki, key in enumerate(gammaparams.keys())})
	if nrand>0:
		pick_rands = np.random.choice(range(len(all_configs)), size = nrand, replace = False).tolist()
		all_configs = [all_configs[i] for i in pick_rands]

	xv_df = []

	# generate splits
	kf = KFold(n_splits=nfolds, random_state = 0)
	splits = [sp for sp in kf.split(data[0])]

	# aux params other than config
	other_params = {
	'data': data,
	'data_ts_gt': data_ts_gt,
	'splits':splits,
	'all_gammas':all_gammas,
	'loss': loss,
	'agg': agg
	}


	if version == "D":
		helper = d_sl_xv_helper
	else:
		helper = c_xv_helper

	if njobs>0:
		pool = mp.Pool(njobs)
		with tqdm(total = len(all_configs)) as pbar:
			for res in pool.imap(partial(helper, **other_params),  all_configs):
				xv_df.append(res)
				pbar.update()
		pool.close()
		pool.join()
	else:
		for combi, config in enumerate(all_configs):
			res = helper(config, **other_params)
			xv_df.append(res)
			if verbose:
				print(res)
	xv_df = pd.concat(xv_df, ignore_index=True)
	return xv_df



def refit_best(version, data, fcr_max, xv_df, metric, loss, agg):

	#---get all the models satisfying the required FCR

	xv_df.fcr.fillna(1e5, inplace = True)
	mfcr = xv_df.fcr.values.copy()

	miw = xv_df[f'{metric}iw'].values.copy()

	# if version == 'D':
	mask = mfcr<= fcr_max
	if np.sum(mask) ==0:
		print(f'min possible FCR = {np.min(mfcr)}')
		mask = mfcr<=np.min(mfcr)



	miw[~mask] = np.max(miw) + 1e10

	#----get optimal
	optid= np.argmin(miw)


	config_opt = xv_df.iloc[optid].copy()

	config_opt = config_opt.to_dict()

	#-----get model
	if version == 'D':
		x, y, w = data
		if len(y.shape) ==2:
			y = y.ravel()

		M = LinearBoundRegression(loss = loss, \
		 agg= agg, standardize=True, **config_opt)
		try:
			M.fit(x, y, w)
		except:
			config_opt['alphau'] = config_opt['alphau'] + 0.1
			config_opt['alphal'] = config_opt['alphal'] + 0.1
			M = LinearBoundRegression(loss = loss, \
		 		agg= agg, standardize=True, **config_opt)
			M.fit(x, y, w)


		return M
	else:
		x, y, t, w = data

		if w is None:
			w_all = None
		else:
			w0, w1 = w[t.ravel()==0], w[t.ravel()==1]
			w0 = 1-w0
			w0 = np.mean(w0)/w0
			w0 = w0/np.sum(w0)

			w1 = np.mean(w1)/w1
			w1 = w1/np.sum(w1)

			w_all = [w0,w1]

		x0, x1 = x[t.ravel()==0], x[t.ravel()==1]
		y0, y1 = y[t.ravel()==0], y[t.ravel()==1]


		M = LinearBoundRegressionCombined(loss = loss, \
		 agg= agg, standardize=True, **config_opt)

		M.fit([x0, x1], [y0, y1], w_all)

		return M





