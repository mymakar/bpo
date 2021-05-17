""" Clean up IST data, undersample untreated older folks"""
import os
import pickle
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from utils import estimate_pscore

MAINDIR = '/main'


def main(args):
	""" main data cleaning function """
	all_cols = [
		'RXHEP', 'RXASP', 'RCONSC', 'SEX', 'AGE', 'RSLEEP', 'RVISINF', 'RSBP',
		'RDEF1', 'RDEF2', 'RDEF3', 'RDEF4', 'RDEF5', 'RDEF6', 'RDEF7', 'RDEF8',
		'STYPE', 'RATRIAL'
	]

	df = pd.read_csv(f'{MAINDIR}/ist_original.CSV', usecols=all_cols, sep=',')
	df.RXHEP[~(df.RXHEP == "N")] = "Y"
	print(df.RXHEP.value_counts())

	print(f'Original n {df.shape[0]}')
	print(f'number heparin only {df[(df.RXHEP=="Y") & (df.RXASP=="N")].shape[0]}')
	print(f'number aspirin only {df[(df.RXHEP=="N") & (df.RXASP=="Y")].shape[0]}')
	print(f'number neither {df[(df.RXHEP=="N") & (df.RXASP=="N")].shape[0]}')
	print(f'number both {df[(df.RXHEP=="Y") & (df.RXASP=="Y")].shape[0]}')

	# ----dropping pilot phase
	df = df[(~df.RATRIAL.isnull())]
	print(f'Non-pilot n {df.shape[0]}')
	print(f'number heparin only {df[(df.RXHEP=="Y") & (df.RXASP=="N")].shape[0]}')
	print(f'number aspirin only {df[(df.RXHEP=="N") & (df.RXASP=="Y")].shape[0]}')
	print(f'number neither {df[(df.RXHEP=="N") & (df.RXASP=="N")].shape[0]}')
	print(f'number both {df[(df.RXHEP=="Y") & (df.RXASP=="Y")].shape[0]}')

	# ----exclude heparin
	df = df[~(df.RXASP == "Y")]

	print(f'After dropping aspiring n {df.shape[0]}')
	print(f'number heparin only {df[(df.RXHEP=="Y")].shape[0]}')
	print(f'number aspirin only {df[(df.RXHEP=="N") & (df.RXASP=="Y")].shape[0]}')
	print(f'number neither {df[(df.RXHEP=="N") & (df.RXASP=="N")].shape[0]}')
	print(f'number both {df[(df.RXHEP=="Y") & (df.RXASP=="Y")].shape[0]}')

	# -----dropping duplicates
	df.drop_duplicates(inplace=True)
	print(f'After dropping duplicates n {df.shape[0]}')
	print(f'number heparin only {df[(df.RXHEP=="Y") & (df.RXASP=="N")].shape[0]}')
	print(f'number aspirin only {df[(df.RXHEP=="N") & (df.RXASP=="Y")].shape[0]}')
	print(f'number neither {df[(df.RXHEP=="N") & (df.RXASP=="N")].shape[0]}')
	print(f'number both {df[(df.RXHEP=="Y") & (df.RXASP=="Y")].shape[0]}')
	df.drop(['RXASP'], axis=1, inplace=True)

	# -----get the treatment
	df['treat'] = np.where(df.RXHEP == "Y", 1, 0)
	df.drop(['RXHEP'], axis=1, inplace=True)

	# ----clean up the variables
	for col in df.columns:
		if col not in ['AGE', 'RSBP', 'treat']:
			if (df[col].unique().all() in ['N', 'Y', 'U']) | (df[col].unique().all() in ['N', 'Y', 'C']) | (df[col].unique().all() in ['N', 'Y']):
				df[col] = np.where(df[col] == 'Y', 1, 0)
			else:
				df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)],
					axis=1)
				df.drop([col], axis=1, inplace=True)

	print(df.columns)
	if args.n_train == 1e6:
		args.n_train, args.n_test = int(df.shape[0] / 2), int(df.shape[0] / 2)
		if args.n_train + args.n_test > df.shape[0]:
			args.n_test = args.n_test - 1

	for repi in range(args.n_reps):
		print(f"=====rep {repi}, third======")
		train_ind = sorted(list(np.random.choice(range(df.shape[0]), args.n_train,
			replace=False)))
		remaining_ind = list(set(range(df.shape[0])) - set(train_ind))
		test_ind = sorted(list(np.random.choice(remaining_ind, args.n_test,
			replace=False)))
		assert len(set(train_ind) & set(test_ind)) == 0

		# -------get the training data
		df_train = df.iloc[train_ind].reset_index(drop=True).copy()
		print(df_train.shape)

		# remove the oldest, untreated patients
		old_untreated = np.where((df_train.AGE > 70) & (df_train.treat == 0))[0]
		others = sorted(list(set(range(df_train.shape[0])) - set(old_untreated)))
		old_keeps = np.random.choice(old_untreated, int(0.3 * len(old_untreated)),
			replace=False).tolist()
		keeps = old_keeps + others

		df_train = df_train.loc[keeps].reset_index(drop=True)

		Y1_train = np.exp(-0.1 * df_train.AGE.ravel())
		Y1_train = MinMaxScaler(feature_range=(2.5, 8)).fit_transform(
			Y1_train.reshape(-1, 1)).ravel()
		Y1_train = Y1_train + np.random.normal(0, .1, Y1_train.shape)

		Y0_train = -np.exp(0.05 * df_train.AGE.ravel())
		Y0_train = MinMaxScaler(feature_range=(0, 2.9)).fit_transform(
			Y0_train.reshape(-1, 1)).ravel()
		Y0_train = Y0_train + np.random.normal(0, .1, Y0_train.shape)

		y_train = df_train['treat'] * Y1_train + (1 - df_train['treat']) * Y0_train

		t_train = df_train.treat.values
		df_train.drop(['treat'], inplace=True, axis=1)
		x_train = df_train.values

		psid = np.random.choice(range(x_train.shape[0]),
			size=int(x_train.shape[0] / 2), replace=False)
		trainid = list(set(range(x_train.shape[0])) - set(psid))

		x_tr = x_train[trainid, :]
		t_tr = t_train[trainid]
		y_tr = y_train[trainid]
		Y1_tr = Y1_train[trainid]
		Y0_tr = Y0_train[trainid]

		x_ps = x_train[psid, :]
		t_ps = t_train[psid]
		y_ps = y_train[psid]
		Y1_ps = Y1_train[psid]
		Y0_ps = Y0_train[psid]
		# ---------get the testing data
		df_test = df.iloc[test_ind].reset_index(drop=True).copy()
		Y1_test = np.exp(-0.1 * df_test.AGE.ravel())
		Y1_test = MinMaxScaler(feature_range=(2.5, 8)).fit_transform(
			Y1_test.reshape(-1, 1)).ravel()
		Y1_test = Y1_test + np.random.normal(0, .1, Y1_test.shape)

		Y0_test = -np.exp(0.05 * df_test.AGE.ravel())
		Y0_test = MinMaxScaler(feature_range=(0, 2.9)).fit_transform(
			Y0_test.reshape(-1, 1)).ravel()
		Y0_test = Y0_test + np.random.normal(0, .1, Y0_test.shape)

		y_test = df_test['treat'] * Y1_test + (1 - df_test['treat']) * Y0_test

		t_test = df_test.treat.values
		df_test.drop(['treat'], inplace=True, axis=1)
		x_test = df_test.values

		data_dict = {
			'x_train': x_tr,
			't_train': t_tr,
			'y_train': y_tr,
			'Y1_train': Y1_tr,
			'Y0_train': Y0_tr,


			'x_ps': x_ps,
			't_ps': t_ps,
			'y_ps': y_ps,
			'Y1_ps': Y1_ps,
			'Y0_ps': Y0_ps,


			'x_test': x_test,
			't_test': t_test,
			'y_test': y_test,
			'Y1_test': Y1_test,
			'Y0_test': Y0_test
		}

		expdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '%s' %args.expname))

		if not os.path.isdir(expdir):
			os.mkdir(expdir)
		if not os.path.isdir(f'{expdir}/output'):
			os.mkdir(f'{expdir}/output')
		if not os.path.isdir(f'{expdir}/results'):
			os.mkdir(f'{expdir}/results')

		pickle.dump(data_dict, open(f'{expdir}/data{repi}.pkl', 'wb'))

		estimate_pscore(f'{expdir}/data{repi}.pkl', f'{expdir}/weights{repi}.csv')


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--expname', '-expname',
		default='ist',
		type=str)

	parser.add_argument('--n_train', '-n_train',
		default=3000,
		help="number of training samples. Note: if set to 1e6, we use the full data",
		type=int)

	parser.add_argument('--n_test', '-n_test',
		default=3000,
		help="number of test samples",
		type=int)

	parser.add_argument('--n_reps', '-n_reps',
		default=20,
		help="number of simulated sample",
		type=int)

	args = parser.parse_args()
	np.random.seed(777)
	main(args)
