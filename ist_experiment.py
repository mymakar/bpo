""" Main IST experiment script"""
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import os, sys, pickle
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser

import bp_main as bp
from bp_main.xvalidation import fcr, iw
import baselines as bl

np.random.seed(20)

MAINDIR = '/main'
expdir = f'{MAINDIR}/ist'
njobs = 15
fcr_max = 0.01
nfolds = 3
p = 1


# ----outcome simulation functions (for sig)

def outcome1(x, t):
	""" Generates outcome under treatment for sigmoid experiment"""
	x0 = x.ravel()
	y = 1 / (1 + np.exp(-t * (x0))) + 2.5
	return y


def outcome0(x, t):
	""" Generates outcome under non-treat for sigmoid experiment"""
	x0 = x.ravel()
	y = 1 / (1 + np.exp(-t * (x0 - 3))) + 1.5
	return y


if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument('--sim', '-sim',
		default='sig',
		choices=['sig', 'hsk'],
		help="simulation name",
		type=str)

	parser.add_argument('--reps', '-reps',
		default="0,2",
		help="comma separated reps",
		type=str)

	args = vars(parser.parse_args())

	rep_min_max = [int(item) for item in args['reps'].split(',')]
	reps = range(rep_min_max[0], rep_min_max[1])
	sim = args['sim']

	# decoupled params

	d_params = {
		'alpha': [0, .1, .5, 1, 2, 5],
		'lamda': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, .5, 1],
		'kernel': ['poly'],
		'p': [p]
	}
	dg_params = {'gamma': [0, 0.01, 0.1, .5, 1, 1.5, 2]}

	# coupled params
	c_params = {
		'alpha1': [0, .1, .5, 1, 2, 5],
		'alpha0': [0, .1, .5, 1, 2, 5],
		'lamda1': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, .5, 1],
		'lamda0': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, .5, 1],
		'kernel': ['poly'],
		'p': [p]
	}
	cg_params = {
		'gamma1': [0, 0.01, 0.1, .5, 1, 1.5, 2],
		'gamma0': [0, 0.01, 0.1, .5, 1, 1.5, 2]}

	for repi in reps:
		data = pickle.load(open(f'{expdir}/data{repi}.pkl', 'rb'))
		p_scores = np.genfromtxt(f'{expdir}/weights{repi}.csv')

		picks = [i for i in range(data['x_train'].shape[0])]
		x_tr = data['x_train'][:, 1][picks].reshape(-1, 1)
		t_tr = data['t_train'][picks]
		p_scores = p_scores[picks]

		# ---extract the treatment and non-treatment
		x0_tr, x1_tr = x_tr[t_tr.ravel()==0], x_tr[t_tr.ravel()==1]
		w0_tr, w1_tr = p_scores[t_tr.ravel()==0].copy(), p_scores[t_tr.ravel()==1].copy()
		w0_tr = 1 - w0_tr

		# ---normalize the weights for the full pop
		wn0_tr = w0_tr.copy()
		wn1_tr = w1_tr.copy()

		wn0_tr = np.mean(wn0_tr) / wn0_tr
		wn0_tr = wn0_tr / np.sum(wn0_tr)

		wn1_tr = np.mean(wn1_tr) / wn1_tr
		wn1_tr = wn1_tr / np.sum(wn1_tr)

		# ----extract test data
		x_ts = data['x_test'][:, 1].reshape(-1, 1)

		# -----scale x
		if sim == "sig":
			xscaler = MinMaxScaler(feature_range=(-10, 10)).fit(x_tr)
			xs_tr = xscaler.transform(x_tr).ravel()
			xs_ts = xscaler.transform(x_ts).ravel()

			# ---set slope
			slope1 = -5
			slope0 = -5
			# ---get Y(1)
			Y1_tr = outcome1(xs_tr, slope1) + np.random.normal(0, .1,
				xs_tr.shape[0])
			Y1_ts = outcome1(xs_ts, slope1) + np.random.normal(0, .1,
				xs_ts.shape[0])

			# ---get Y(0)
			Y0_tr = outcome0(xs_tr, slope0)+ np.random.normal(0, .1,
				xs_tr.shape[0])
			Y0_ts = outcome0(xs_ts, slope0) + np.random.normal(0, .1,
				xs_ts.shape[0])
		else:

			# ---Y1
			Y1_tr = np.exp(-0.1 * x_tr.ravel())
			y1scaler = MinMaxScaler(feature_range=(2.5, 8)).fit(
				Y1_tr.reshape(-1, 1))
			Y1_tr = y1scaler.transform(Y1_tr.reshape(-1, 1)).ravel()
			Y1_tr = Y1_tr + np.random.normal(0, .1, x_tr.shape[0])

			Y1_ts = np.exp(-0.1 * x_ts.ravel())
			Y1_ts = y1scaler.transform(Y1_ts.reshape(-1, 1)).ravel()
			Y1_ts = Y1_ts + np.random.normal(0, .1, x_ts.shape[0])

			# ---Y0
			Y0_tr = -np.exp(0.1 * x_tr.ravel())
			yscaler = MinMaxScaler(feature_range=(0.1, 3)).fit(
				Y0_tr.reshape(-1, 1))
			Y0_tr = yscaler.transform(Y0_tr.reshape(-1, 1)).ravel()
			Y0_tr = Y0_tr + np.random.normal(0, .1, Y0_tr.shape)
			Y0_tr[Y0_tr < 0] = 0

			Y0_ts = -np.exp(0.1 * x_ts.ravel())
			Y0_ts = yscaler.transform(Y0_ts.reshape(-1, 1)).ravel()
			Y0_ts = Y0_ts + np.random.normal(0, .1, Y0_ts.shape)
			Y0_ts[Y0_ts < 0] = 0

		# ---get the observed outcome for the training data
		y_tr = Y1_tr * t_tr + Y0_tr * (1 - t_tr)
		y0_tr, y1_tr = y_tr[t_tr.ravel() == 0], y_tr[t_tr.ravel() == 1]

		# ---------------------------------#
		# --------Training-----------------#
		# ---------------------------------#

		# ----------Y0 BPD SQ

		xv_bpd_sq_0 = bp.bp_xv(version='D', nfolds=3,
			data=(x0_tr.copy(), y0_tr.copy(), w0_tr.copy()),
			params=d_params, gammaparams=dg_params, loss='square', agg='mean',
			njobs=njobs, tried_params=None, verbose=False, temp_file=None,
			nrand=0)

		M_bpd_sq_0 = bp.refit_best('D',
			data=(x0_tr.copy(), y0_tr.copy(), wn0_tr.copy()), fcr_max=fcr_max,
			xv_df=xv_bpd_sq_0, metric='mean', loss='square', agg='mean')

		l0_bpd_sq, u0_bpd_sq = M_bpd_sq_0.predict(x_ts)

		fcr0_bpd_sq = fcr(Y0_ts, l0_bpd_sq, u0_bpd_sq)
		iw0_bpd_sq = iw(l0_bpd_sq, u0_bpd_sq)
		iwm0_bpd_sq = iw(l0_bpd_sq, u0_bpd_sq, op='max')

		# ---------BPC SQ

		xv_bpc_sq = bp.bp_xv(version='C', nfolds=3,
			data=(x_tr.copy(), y_tr.copy(), t_tr.copy(), p_scores.copy()),
			params=c_params, gammaparams=cg_params, loss='square',
			agg='mean', njobs=njobs, tried_params=None, verbose=False,
			temp_file=None, nrand=0)

		M_bpc_sq = bp.refit_best('C',
			data=(x_tr.copy(), y_tr.copy(), t_tr.copy(), p_scores.copy()),
			fcr_max=fcr_max / 2, xv_df=xv_bpc_sq, metric='mean',
			loss='square', agg='mean')

		l0_bpc_sq, u0_bpc_sq, l1_bpc_sq, u1_bpc_sq = M_bpc_sq.predict(x_ts)

		fcr0_bpc_sq = fcr(Y0_ts, l0_bpc_sq, u0_bpc_sq)
		iw0_bpc_sq = iw(l0_bpc_sq, u0_bpc_sq)
		iwm0_bpc_sq = iw(l0_bpc_sq, u0_bpc_sq, op='max')

		fcr1_bpc_sq = fcr(Y1_ts, l1_bpc_sq, u1_bpc_sq)
		iw1_bpc_sq = iw(l1_bpc_sq, u1_bpc_sq)
		iwm1_bpc_sq = iw(l1_bpc_sq, u1_bpc_sq, op='max')

		# -------Y1, BPD, SQ

		xv_bpd_sq_1 = bp.bp_xv(version='D', nfolds=3,
			data=(x1_tr.copy(), y1_tr.copy(), w1_tr.copy()),
			params=d_params, gammaparams=dg_params, loss='square', agg='mean',
			njobs=njobs, tried_params=None, verbose=False, temp_file=None,
			nrand=0)

		M_bpd_sq_1 = bp.refit_best('D',
			data=(x1_tr.copy(), y1_tr.copy(), wn1_tr.copy()), fcr_max=fcr_max,
			xv_df=xv_bpd_sq_1, metric='mean', loss='square', agg='mean')

		l1_bpd_sq, u1_bpd_sq = M_bpd_sq_1.predict(x_ts)

		fcr1_bpd_sq = fcr(Y1_ts, l1_bpd_sq, u1_bpd_sq)
		iw1_bpd_sq = iw(l1_bpd_sq, u1_bpd_sq)
		iwm1_bpd_sq = iw(l1_bpd_sq, u1_bpd_sq, op='max')

		# ----Y1 BPD, max

		xv_bpd_mx_1 = bp.bp_xv(version='D', nfolds=3,
			data=(x1_tr.copy(), y1_tr.copy(), w1_tr.copy()),
			params=d_params, gammaparams=dg_params, loss='linear', agg='max',
			njobs=njobs, tried_params=None, verbose=False, temp_file=None,
			nrand=0)

		M_bpd_mx_1 = bp.refit_best('D',
			data=(x1_tr.copy(), y1_tr.copy(), wn1_tr.copy()), fcr_max=fcr_max,
			xv_df=xv_bpd_mx_1, metric='max', loss='linear', agg='max')

		l1_bpd_mx, u1_bpd_mx = M_bpd_mx_1.predict(x_ts)

		fcr1_bpd_mx = fcr(Y1_ts, l1_bpd_mx, u1_bpd_mx)
		iw1_bpd_mx = iw(l1_bpd_mx, u1_bpd_mx)
		iwm1_bpd_mx = iw(l1_bpd_mx, u1_bpd_mx, op='max')

		# ---Y1 BPD LIN

		xv_bpd_li_1 = bp.bp_xv(version='D', nfolds=3,
			data=(x1_tr.copy(), y1_tr.copy(), w1_tr.copy()),
			params=d_params, gammaparams=dg_params, loss='linear', agg='mean',
			njobs=njobs, tried_params=None, verbose=False, temp_file=None,
			nrand=0)

		M_bpd_li_1 = bp.refit_best('D',
			data=(x1_tr.copy(), y1_tr.copy(), wn1_tr.copy()), fcr_max=fcr_max,
			xv_df=xv_bpd_li_1, metric='mean', loss='linear', agg='mean')

		l1_bpd_li, u1_bpd_li = M_bpd_li_1.predict(x_ts)

		fcr1_bpd_li = fcr(Y1_ts, l1_bpd_li, u1_bpd_li)
		iw1_bpd_li = iw(l1_bpd_li, u1_bpd_li)
		iwm1_bpd_li = iw(l1_bpd_li, u1_bpd_li, op='max')

		# -----Y1, QR
		# NOTE: QR does not have the gamma parameter. So we set Gamma = 0
		# to get the classic QR
		xv_qr_1 = xv_bpd_li_1.copy()
		xv_qr_1 = xv_qr_1[((xv_qr_1.gammal == 0) & (xv_qr_1.gammau == 0))]
		M_qr_1 = bp.refit_best('D',
			data=(x1_tr.copy(), y1_tr.copy(), wn1_tr.copy()), fcr_max=fcr_max,
			xv_df=xv_qr_1, metric='mean', loss='linear', agg='mean')

		l1_qr, u1_qr = M_qr_1.predict(x_ts)

		fcr1_qr = fcr(Y1_ts, l1_qr, u1_qr)
		iw1_qr = iw(l1_qr, u1_qr)
		iwm1_qr = iw(l1_qr, u1_qr, op='max')

		# ---------Fit ERMinimizer
		# split data into 2
		i1 = np.random.choice(range(x1_tr.shape[0]),
			size=int(x1_tr.shape[0] / 2), replace=False)
		i2 = list(set(range(x1_tr.shape[0])) - set(i1))

		ps1 = p_scores[t_tr.ravel() == 1].copy()
		x1_ft = x1_tr[i1, :].copy()
		w1_ft = ps1[i1].copy()
		y1_ft = y1_tr[i1].copy()

		x1_cnf = x_tr[i2, :].copy()
		w1_cnf = ps1[i2].copy()
		y1_cnf = y_tr[i2].copy()

		wn1_ft = np.mean(wn1_tr) / wn1_tr
		wn1_ft = wn1_tr / np.sum(wn1_tr)

		# xvalidate
		params = (['alpha', 'p'], ([0, 1e-5, 1e-2, .1, 1, 2, 3, 4, 5], [p]))
		xv_erm, _ = bl.reg_sl_xv("kr", nfolds=nfolds,
			data=(x1_ft.copy(), y1_ft.copy(), w1_ft.copy()), params=params)

		# get best
		xv_erm['mse'] = np.mean(xv_erm[[f'mse_{i}' for i in range(nfolds)]],
			axis=1)
		best_erm = xv_erm[(xv_erm.mse == np.min(xv_erm.mse))]
		print(best_erm)

		# ----conformal intervals

		Mcnf = bl.UnconstrainedRegression("kr", gammau=0, gammal=0,
			alpha=best_erm['alpha'], kernel="poly", p=int(best_erm['p']))
		Mcnf.fit(x1_ft, y1_ft, wn1_ft)
		Mcnf.fit_conformal(fcr_max, x1_cnf, y1_cnf)
		mcnf_l1, mcnf_u1 = Mcnf.predict(x_ts)
		fcr1_mcnf = fcr(Y1_ts, mcnf_l1, mcnf_u1)
		iw1_mcnf = iw(mcnf_l1, mcnf_u1)
		iwm1_mcnf = iw(mcnf_l1, mcnf_u1, op='max')

		# ----gamma interval

		Mkr = bl.UnconstrainedRegression("kr", gammau=0, gammal=0,
			alpha=best_erm['alpha'], kernel="poly", p=int(best_erm['p']))

		Mkr.fit(x1_ft.copy(), y1_ft.copy(), wn1_ft.copy())
		yh_cnf = Mkr.predict(x1_cnf, pred_type="est")

		gammaparams = {
			'gammal': [0, 0.0001, 0.01, 0.1, .2, 0.3, 0.4, 0.5] + np.linspace(
				.5, 2, 10).tolist(),
			'gammau': [0, 0.0001, 0.01, 0.1, .2, 0.3, 0.4, 0.5] + np.linspace(
				.5, 2, 10).tolist()
		}

		xv_gint = bl.gint_sl_xv(gammaparams, (y1_cnf, w1_cnf), yh_cnf)

		if np.min(xv_gint.fcr) > fcr_max:
			vld_fcr = xv_gint[(xv_gint.fcr <= np.min(
				xv_gint.fcr))].reset_index()
		else:
			vld_fcr = xv_gint[(xv_gint.fcr <= fcr_max)].reset_index()
		best_gint = vld_fcr[(
			vld_fcr.meaniw == np.min(vld_fcr.meaniw))].reset_index(drop=True)
		if best_gint.shape[0] > 1:
			gintid = np.random.choice(best_gint.index, size=1, replace=False)
			best_gint = best_gint.iloc[gintid]

		print(best_gint)
		Mkr.set_gamma(gammau=best_gint['gammau'].values,
			gammal=best_gint['gammal'].values)

		kr_l1, kr_u1 = Mkr.predict(x_ts)
		fcr1_kr = fcr(Y1_ts, kr_l1, kr_u1)
		iw1_kr = iw(kr_l1, kr_u1)
		iwm1_kr = iw(kr_l1, kr_u1, op='max')

		# --------------------------------------#
		# --------results-----------------------#
		# --------------------------------------#

		res = {}

		res['R0'] = pd.DataFrame([
			{'Model': 'Decoup', 'FCR': fcr0_bpd_sq, 'Mean IW': iw0_bpd_sq,
				'Max IW': iwm0_bpd_sq},
			{'Model': 'Coup', 'FCR': fcr0_bpc_sq, 'Mean IW': iw0_bpc_sq,
				'Max IW': iwm0_bpc_sq}
		])
		print('=====Results for Y0=======')
		print(res['R0'])
		# -------------Y0 PLOTS

		# rc('text', usetex=True)
		rc('font', family='serif', size=25)

		plt.figure(figsize=(7, 6))
		x_plt = np.expand_dims(np.arange(np.min(x_ts), np.max(x_ts), 1), 1)

		l0_sqi_plt, u0_sqi_plt = M_bpd_sq_0.predict(x_plt)
		l0_sq_plt, u0_sq_plt, _, _ = M_bpc_sq.predict(x_plt)

		plt.scatter(x_ts, Y0_ts, color='black', s=5)

		plt.plot(x_plt, l0_sqi_plt, c='teal', label='BP-D-L2',
			marker='s', markevery=5, linewidth=3)
		plt.plot(x_plt, u0_sqi_plt, c='teal', marker='s', markevery=5,
			linewidth=3)

		plt.plot(x_plt, l0_sq_plt, c='darkmagenta', label='BP-C-L2',
			linewidth=3)
		plt.plot(x_plt, u0_sq_plt, c='darkmagenta', linewidth=3)

		plt.axhspan(2, 3, alpha=0.2, label="Normal Range")

		plt.xlabel(r'Age')
		plt.ylabel(r'INR under control, $Y(0)$')

		plt.legend(prop={'size': 17}, bbox_to_anchor=(-0.01, .01),
			loc="lower left")
		plt.tight_layout()

		plt.savefig(f'{expdir}/results/{sim}/ist_y0_rep{repi}.pdf')

		preds0 = {
			'x_plt': x_plt,
			'l0_sqi_plt': l0_sqi_plt,
			'u0_sqi_plt': u0_sqi_plt,
			'l0_sq_plt': l0_sq_plt,
			'u0_sq_plt': u0_sq_plt,
			'x_ts': x_ts,
			'Y0_ts': Y0_ts
		}

		res['R1'] = pd.DataFrame([
			{'Model': 'Max', 'FCR': fcr1_bpd_mx, 'Mean IW': iw1_bpd_mx,
				'Max IW': iwm1_bpd_mx},
			{'Model': 'Linear', 'FCR': fcr1_bpd_li, 'Mean IW': iw1_bpd_li,
				'Max IW': iwm1_bpd_li},
			{'Model': 'QR', 'FCR': fcr1_qr, 'Mean IW': iw1_qr,
				'Max IW': iwm1_qr},
			{'Model': 'Conf ', 'FCR': fcr1_mcnf, 'Mean IW': iw1_mcnf,
				'Max IW': iwm1_mcnf},
			{'Model': 'KR ', 'FCR': fcr1_kr, 'Mean IW': iw1_kr,
				'Max IW': iwm1_kr},
			{'Model': 'comb', 'FCR': fcr1_bpc_sq, 'Mean IW': iw1_bpc_sq,
				'Max IW': iwm1_bpc_sq},
			{'Model': 'SQ', 'FCR': fcr1_bpd_sq, 'Mean IW': iw1_bpd_sq,
				'Max IW': iwm1_bpd_sq}
		])

		print('=====Results for Y1=======')
		print(res['R1'])

		rc('font', family='serif', size=25)

		plt.figure(figsize=(7, 6))

		x_plt = np.expand_dims(np.arange(np.min(x_ts), np.max(x_ts), 1), 1)
		if sim == "sig":
			x1_plt = xscaler.transform(x_plt).ravel()
		else:
			x1_plt = x_plt

		l1_mxi_plt, u1_mxi_plt = M_bpd_mx_1.predict(x_plt)
		l1_li_plt, u1_li_plt = M_bpd_li_1.predict(x_plt)
		l1_qr_plt, u1_qr_plt = M_qr_1.predict(x_plt)
		l1_krr_plt, u1_krr_plt = Mkr.predict(x_plt)
		l1_cnf_plt, u1_cnf_plt = Mcnf.predict(x_plt)
		l1_sq_plt, u1_sq_plt = M_bpd_sq_1.predict(x_plt)
		_, _, cl1_sq_plt, cu1_sq_plt = M_bpc_sq.predict(x_plt)

		plt.scatter(x_ts, Y1_ts, color='black', s=5)
		plt.plot(x_plt, l1_qr_plt, c='green', label='QR',
			linewidth=3, markevery=5, marker='o')
		plt.plot(x_plt, u1_qr_plt, c='green', linewidth=3,
			markevery=5, marker='o')

		plt.plot(x_plt, l1_krr_plt, c='orange', label='KR-$\gamma$',
			linewidth=5)
		plt.plot(x_plt, u1_krr_plt, c='orange', linewidth=5)

		plt.plot(x_plt, l1_cnf_plt, c='purple', label='KR-MI', linestyle=":",
			linewidth=5)
		plt.plot(x_plt, u1_cnf_plt, c='purple', linestyle=":", linewidth=5)

		plt.plot(x_plt, l1_mxi_plt, c='blue', label=r'BP-D-L$\infty$',
			linestyle="--", linewidth=3)
		plt.plot(x_plt, u1_mxi_plt, c='blue', linestyle="--", linewidth=3)

		plt.plot(x_plt, l1_sq_plt, c='r', label='BP-D-L2', linewidth=3)
		plt.plot(x_plt, u1_sq_plt, c='r', linewidth=3)
		plt.axhspan(2, 3, alpha=0.2, label="Normal\n Range")

		plt.xlabel(r'Age')
		plt.ylabel(r'INR under treatment, $Y(1)$')

		plt.legend(ncol=2, columnspacing=0.2, prop={'size': 17},
			bbox_to_anchor=(.4, 1.02), loc="upper left")
		plt.tight_layout()

		plt.savefig(f'{expdir}/results/{sim}/ist_y1_rep{repi}.pdf')
		plt.clf()

		preds1 = {
			'x_plt': x_plt,
			'l1_mxi_plt': l1_mxi_plt,
			'u1_mxi_plt': u1_mxi_plt,
			'l1_li_plt': l1_li_plt,
			'u1_li_plt': u1_li_plt,
			'l1_qr_plt': l1_qr_plt,
			'u1_qr_plt': u1_qr_plt,
			'l1_krr_plt': l1_krr_plt,
			'u1_krr_plt': u1_krr_plt,
			'l1_cnf_plt': l1_cnf_plt,
			'u1_cnf_plt': u1_cnf_plt,
			'l1_sq_plt': l1_sq_plt,
			'u1_sq_plt': u1_sq_plt,
			'cl1_sq_plt': cl1_sq_plt,
			'cu1_sq_plt': cu1_sq_plt,
			'x_ts': x_ts,
			'Y1_ts': Y1_ts
		}

		pickle.dump(res, open(f'{expdir}/results/{sim}/res_rep{repi}.pkl', 'wb'))
		pickle.dump(preds0, open(f'{expdir}/results/{sim}/preds0_rep{repi}.pkl', 'wb'))
		pickle.dump(preds1, open(f'{expdir}/results/{sim}/preds1_rep{repi}.pkl', 'wb'))
