""" Propensity score estimation utility files """
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def sigmoid(x):
	""" sigmoid function"""
	return 1 / (1 + np.exp(-x))


def estimate_pscore(expdir, logistic=True):
	""" Train propensity model and get estimates for datasets"""
	data = pickle.load(open(f'{expdir}/data.pkl', 'rb'))
	x_ps, t_ps = data['x_ps'], data['t_ps']

	if logistic:
		estimator = LogisticRegression(random_state=0, solver='lbfgs')
		param_grid = {'C': np.logspace(-4, -4, 100).tolist()}
	else:
		estimator = RandomForestClassifier()
		param_grid = {
			'max_depth': [1, 2, 3, 4, 5],
			'n_estimators': [200, 500, 1000]
		}

	grid_search = GridSearchCV(estimator=estimator,
		param_grid=param_grid,
		cv=KFold(n_splits=3), n_jobs=5, refit=True, verbose=0)
	grid_search.fit(x_ps, t_ps.ravel())
	clf = grid_search.best_estimator_

	# ----train the train dataset
	pred = clf.predict_proba(data['x_tr'])[:, 1]
	pred_old = np.copy(pred)
	pred95 = np.quantile(pred, 0.95)
	pred05 = np.quantile(pred, 0.05)
	pred[pred > pred95] = pred95
	pred[pred < pred05] = pred05
	print(f'{pred95:.3f}, {np.max(pred):.3f}, {pred05:.3f}, {np.min(pred05):.3f}, {np.mean(pred_old):.3f}, {np.mean(pred):.3f}')
	fpr, tpr, _ = metrics.roc_curve(data['t_tr'], pred, pos_label=1)
	print(f'pscore accuracy: {metrics.auc(fpr, tpr)}')
	np.savetxt(f'{expdir}/weights.csv', pred, delimiter=",")

	# ----prediction for the train (not calibration) dataset
	pred = clf.predict_proba(data['x_tr_tr'])[:, 1]
	pred_old = np.copy(pred)
	pred95 = np.quantile(pred, 0.95)
	pred05 = np.quantile(pred, 0.05)
	pred[pred > pred95] = pred95
	pred[pred < pred05] = pred05
	print(f'{pred95:.3f}, {np.max(pred):.3f}, {pred05:.3f}, {np.min(pred05):.3f}, {np.mean(pred_old):.3f}, {np.mean(pred):.3f}')
	fpr, tpr, _ = metrics.roc_curve(data['t_tr_tr'], pred, pos_label=1)
	print(f'pscore accuracy: {metrics.auc(fpr, tpr)}')
	np.savetxt(f'{expdir}/weights_tr_tr.csv', pred, delimiter=",")

	# ----prediction for the calibration dataset
	pred = clf.predict_proba(data['x_tr_cal'])[:, 1]
	pred_old = np.copy(pred)
	pred95 = np.quantile(pred, 0.95)
	pred05 = np.quantile(pred, 0.05)
	pred[pred > pred95] = pred95
	pred[pred < pred05] = pred05
	print(f'{pred95:.3f}, {np.max(pred):.3f}, {pred05:.3f}, {np.min(pred05):.3f}, {np.mean(pred_old):.3f}, {np.mean(pred):.3f}')
	fpr, tpr, _ = metrics.roc_curve(data['t_tr_cal'], pred, pos_label=1)
	print(f'pscore accuracy: {metrics.auc(fpr, tpr)}')
	np.savetxt(f'{expdir}/weights_tr_cal.csv', pred, delimiter=",")
