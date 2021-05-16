""" Main bound estimation models """
import gurobipy as grb
import numpy as np

from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import RBFSampler


class LinearBoundRegression():
	""" Linear Bound Regression """

	def __init__(self, loss, agg,
		gammau, gammal, lamdau, lamdal,
		alphau, alphal, kernel,
		p=None, sig=None, standardize=True, **unused_kwargs):

		""" Constructor """

		self.loss = loss
		self.agg = agg

		self.lamdau = lamdau
		self.lamdal = lamdal

		self.alphau = alphau
		self.alphal = alphal

		self.gammau = gammau
		self.gammal = gammal

		self.kernel = kernel
		self.p = p
		self.sig = sig
		self.standardize = standardize

	def fit(self, x_or, y, w=None):
		""" Fits upper and lower bounds on p(y|x) """

		if self.standardize:
			xselector = VarianceThreshold(threshold=.1).fit(x_or)
			temp_x = xselector.transform(x_or)
			xscaler = StandardScaler().fit(temp_x)
			self.xscaler = lambda x: xscaler.transform(xselector.transform(x))
			x = self.xscaler(x_or)
		else:
			x = x_or.copy()

		if self.kernel == 'linear':
			self.kernel_fit = lambda x: x
			x = self.kernel_fit(x)

		elif self.kernel == 'poly':
			if self.p is None:
				raise ValueError('Need polynomial value')

			self.kernel_fit = lambda x: np.hstack([x**i for i in range(1, self.p + 1)])
			x = self.kernel_fit(x)

		elif self.kernel == 'rbf':
			if self.sig is None:
				raise ValueError('Need Length scale value')
			self.x_tr = x.copy()
			self.kernel_fit = lambda x_ts: RBF(length_scale=self.sig).__call__(x_ts,
				self.x_tr)
			x = self.kernel_fit(x)

		elif self.kernel == 'rbf_approx':
			if self.sig is None:
				raise ValueError('Need Length scale value')

			rbf_fit = RBFSampler(gamma=1 / self.sig, n_components=50).fit(x.copy())
			self.kernel_fit = lambda x_ts: rbf_fit.transform(x_ts)
			x = self.kernel_fit(x)

		n, d = x.shape[0], x.shape[1]
		mdl = grb.Model("qp")
		mdl.ModelSense = 1
		mdl.setParam('OutputFlag', False)
		mdl.reset()

		L = 1e5
		us = [mdl.addVar(name="u%d" % i, lb=-L, ub=L) for i in range(n)]
		ls = [mdl.addVar(name="l%d" % i, lb=-L, ub=L) for i in range(n)]
		bsU = [mdl.addVar(name="bu%d" % i, lb=-L, ub=L) for i in range(d + 1)]
		bsL = [mdl.addVar(name="bl%d" % i, lb=-L, ub=L) for i in range(d + 1)]
		rUs = [mdl.addVar(name="ru%d" % i, lb=0, ub=L) for i in range(n)]
		rLs = [mdl.addVar(name="rl%d" % i, lb=0, ub=L) for i in range(n)]

		slackU = 0
		slackL = 0

		if w is None:
			w = np.ones(n) / n

		obj_terms = []
		for i in range(n):
			mdl.addConstr(us[i] >= ls[i])

			mdl.addConstr(us[i] == np.dot(x[i, ], bsU[:d]) + bsU[-1])
			mdl.addConstr(ls[i] == np.dot(x[i, ], bsL[:d]) + bsL[-1])

			mdl.addConstr(rUs[i] >= y[i] - us[i])
			mdl.addConstr(rLs[i] >= ls[i] - y[i])

			slackU += w[i] * rUs[i]
			slackL += w[i] * rLs[i]

			if self.loss == 'square':
				obj_terms.append(w[i] * (us[i] - ls[i]) * (us[i] - ls[i]))
			elif self.loss == 'linear':
				if self.agg == 'max':
					obj_terms.append((us[i] - ls[i]))
				else:
					obj_terms.append(w[i] * (us[i] - ls[i]))

			else:
				raise Exception('Unrecognized loss: %s' % self.loss)

		if self.agg == 'max':
			o = mdl.addVar(name="o", lb=-L, ub=L)
			os = []
			for i in range(n):
				oi = mdl.addVar(name="o%d" % i, lb=-L, ub=L)
				mdl.addConstr(oi == obj_terms[i])
				os += [oi]
			mdl.addConstr(o == grb.max_(os))
			obj = o
		else:
			obj = grb.quicksum(obj_terms)

		# ----add the values of the objectives
		obj_reg_u, obj_reg_l = 0, 0
		for k in range(d):
			obj_reg_u += bsU[k] * bsU[k]
			obj_reg_l += bsL[k] * bsL[k]

		obj_reg = self.alphau * obj_reg_u + self.alphal * obj_reg_l

		mdl.addConstr(slackU <= self.lamdau)
		mdl.addConstr(slackL <= self.lamdal)
		obj_f = obj + obj_reg

		mdl.setObjective(obj_f)
		mdl.optimize()

		self.bu = np.array([bsU[j].x for j in range(d + 1)])
		self.bl = np.array([bsL[j].x for j in range(d + 1)])

		# print(obj.getValue(), obj_slack.getValue())

		return self

	def predict(self, x):
		""" Predicts lower and upper bounds at a point x """
		if self.standardize:
			x = self.xscaler(x)

		x = self.kernel_fit(x)

		yhu = np.dot(x, self.bu[:x.shape[1]]) + self.bu[-1] + self.gammau
		yhl = np.dot(x, self.bl[:x.shape[1]]) + self.bl[-1] - self.gammal

		return yhl, yhu

	def set_params(self, gammau=None, gammal=None):
		""" Sets the parameters of the model """
		if gammau is not None:
			self.gammau = gammau
		if gammal is not None:
			self.gammal = gammal


class LinearBoundRegressionCombined():
	""" Linear Bound Regression combined treatment groups"""

	def __init__(self, loss, agg,
		gammal0, gammal1, gammau0, gammau1,
		lamdal0, lamdal1, lamdau0, lamdau1,
		alphal0, alphal1, alphau0, alphau1,
		kernel, p=None, sig=None, standardize=True, **unused_kwargs):

		""" Constructor """

		self.loss = loss
		self.agg = agg

		self.lamdal0 = lamdal0
		self.lamdau0 = lamdau0
		self.lamdal1 = lamdal1
		self.lamdau1 = lamdau1

		self.alphal0 = alphal0
		self.alphau0 = alphau0
		self.alphal1 = alphal1
		self.alphau1 = alphau1

		self.gammau0 = gammau0
		self.gammal0 = gammal0
		self.gammau1 = gammau1
		self.gammal1 = gammal1

		self.kernel = kernel
		self.p = p
		self.sig = sig
		self.standardize = standardize

	def fit(self, x, y, w=None):
		""" Fits upper and lower bounds on p(y|x)
		Args:
			x, y are lists with control groups first
		"""

		# -----preprocessing
		if self.standardize:

			x0selector = VarianceThreshold(threshold=.1).fit(x[0])
			temp_x0 = x0selector.transform(x[0])
			x0scaler = StandardScaler().fit(temp_x0)
			self.x0scaler = lambda x: x0scaler.transform(x0selector.transform(x))

			x1selector = VarianceThreshold(threshold=.1).fit(x[1])
			temp_x1 = x1selector.transform(x[1])
			x1scaler = StandardScaler().fit(temp_x1)
			self.x1scaler = lambda x: x1scaler.transform(x1selector.transform(x))

			x00 = self.x0scaler(x[0])
			x01 = self.x1scaler(x[0])
			x11 = self.x1scaler(x[1])
			x10 = self.x0scaler(x[1])

		else:
			x00, x01 = x[0], x[0]
			x11, x10 = x[1], x[1]

		if self.kernel == 'linear':
			self.kernel_fit = lambda x: x
			x00 = self.kernel_fit(x00)
			x01 = self.kernel_fit(x01)
			x11 = self.kernel_fit(x11)
			x10 = self.kernel_fit(x10)

		elif self.kernel == 'poly':
			if self.p is None:
				raise ValueError('Need polynomial value')

			self.kernel_fit = lambda x: np.hstack([x**i for i in range(1, self.p + 1)])
			x00 = self.kernel_fit(x00)
			x01 = self.kernel_fit(x01)
			x11 = self.kernel_fit(x11)
			x10 = self.kernel_fit(x10)

		elif self.kernel == 'rbf':
			if self.sig is None:
				raise ValueError('Need Length scale value')
			self.x0_tr = x00.copy()
			self.x1_tr = x11.copy()

			self.kernel_fit = lambda x, tg: RBF(length_scale=self.sig).__call__(x,
				self.x1_tr) if tg == 1 else \
				RBF(length_scale=self.sig).__call__(x, self.x0_tr)

			x00 = self.kernel_fit(x00, tg=0)
			x01 = self.kernel_fit(x01, tg=1)
			x11 = self.kernel_fit(x11, tg=1)
			x10 = self.kernel_fit(x10, tg=0)

		elif self.kernel == 'rbf_approx':
			if self.sig is None:
				raise ValueError('Need Length scale value')

			self.x0_tr = x00.copy()
			self.x1_tr = x11.copy()

			self.rbf_approx1 = RBFSampler(gamma=1 / self.sig, n_components=100,
				random_state=0).fit(self.x1_tr)
			self.rbf_approx0 = RBFSampler(gamma=1 / self.sig, n_components=100,
				random_state=0).fit(self.x0_tr)

			self.kernel_fit = lambda x, tg: self.rbf_approx1.transform(x) if tg == 1 else \
				self.rbf_approx0.transform(x)

			x00 = self.kernel_fit(x00, tg=0)
			x01 = self.kernel_fit(x01, tg=1)
			x11 = self.kernel_fit(x11, tg=1)
			x10 = self.kernel_fit(x10, tg=0)

		n1, d1 = x11.shape[0], x11.shape[1]
		n0, d0 = x00.shape[0], x00.shape[1]
		y0 = y[0]
		y1 = y[1]

		n = n1 + n0

		mdl = grb.Model("cqp")
		mdl.ModelSense = 1
		mdl.setParam('OutputFlag', False)
		mdl.reset()
		L = 1e5

		u0 = [mdl.addVar(name="u0_%d" % i, lb=-L, ub=L) for i in range(n)]
		l0 = [mdl.addVar(name="l0_%d" % i, lb=-L, ub=L) for i in range(n)]

		u1 = [mdl.addVar(name="u1_%d" % i, lb=-L, ub=L) for i in range(n)]
		l1 = [mdl.addVar(name="l1_%d" % i, lb=-L, ub=L) for i in range(n)]

		bU0 = [mdl.addVar(name="bu0_%d" % i, lb=-L, ub=L) for i in range(d0 + 1)]
		bL0 = [mdl.addVar(name="bl0_%d" % i, lb=-L, ub=L) for i in range(d0 + 1)]

		bU1 = [mdl.addVar(name="bu1_%d" % i, lb=-L, ub=L) for i in range(d1 + 1)]
		bL1 = [mdl.addVar(name="bl1_%d" % i, lb=-L, ub=L) for i in range(d1 + 1)]

		rUs = [mdl.addVar(name="ru%d" % i, lb=0, ub=L) for i in range(n)]
		rLs = [mdl.addVar(name="rl%d" % i, lb=0, ub=L) for i in range(n)]

		slackU1 = 0
		slackL1 = 0

		slackU0 = 0
		slackL0 = 0

		if w is None:
			w0, w1= np.ones(n0) / n0, np.ones(n1) / n1
		else:
			w0 = w[0]
			w1 = w[1]

		obj_terms = []
		for i in range(n):
			mdl.addConstr(u1[i] >= l1[i])
			mdl.addConstr(u0[i] >= l0[i])

		for i in range(n0):

			mdl.addConstr(u1[i] == np.dot(x01[i, ], bU1[:d1]) + bU1[-1])
			mdl.addConstr(l1[i] == np.dot(x01[i, ], bL1[:d1]) + bL1[-1])

			mdl.addConstr(u0[i] == np.dot(x00[i, ], bU0[:d0]) + bU0[-1])
			mdl.addConstr(l0[i] == np.dot(x00[i, ], bL0[:d0]) + bL0[-1])

			mdl.addConstr(rUs[i] >= y0[i] - u0[i])
			mdl.addConstr(rLs[i] >= l0[i] - y0[i])

			slackU0 += w0[i] * rUs[i]
			slackL0 += w0[i] * rLs[i]

			if self.loss == 'square':
				obj_terms.append(w0[i] * ((u0[i] - l0[i]) * (u0[i] - l0[i]) + (u1[i] - l1[i]) * (u1[i] - l1[i])))
			elif self.loss == 'linear':
				if self.agg == "max":
					obj_terms.append(((u0[i] - l0[i]) + (u1[i] - l1[i])))
				else:
					obj_terms.append(w0[i]*((u0[i] - l0[i])+ (u1[i] - l1[i])))
			else:
				raise Exception('Unrecognized loss: %s' % self.loss)

		for i in range(n0, n1+n0):

			mdl.addConstr(u1[i] == np.dot(x11[i - n0, ], bU1[:d1]) + bU1[-1])
			mdl.addConstr(l1[i] == np.dot(x11[i - n0, ], bL1[:d1]) + bL1[-1])

			mdl.addConstr(u0[i] == np.dot(x10[i - n0, ], bU0[:d0]) + bU0[-1])
			mdl.addConstr(l0[i] == np.dot(x10[i - n0, ], bL0[:d0]) + bL0[-1])

			mdl.addConstr(rUs[i] >= y1[i - n0] - u1[i])
			mdl.addConstr(rLs[i] >= l1[i] - y1[i - n0])

			slackU1 += w1[i - n0] * rUs[i]
			slackL1 += w1[i - n0] * rLs[i]

			if self.loss == 'square':
				obj_terms.append(w1[i - n0] * ((u1[i] - l1[i]) * (u1[i] - l1[i])))

			elif self.loss == 'linear':
				if self.agg == "max":
					obj_terms.append(((u1[i] - l1[i]) + (u0[i] - l0[i])))
				else:
					obj_terms.append(w1[i - n0] * ((u1[i] - l1[i]) + (u0[i] - l0[i])))

			else:
				raise Exception('Unrecognized loss: %s' % self.loss)

		if self.agg == 'max':
			o = mdl.addVar(name="o", lb=-L, ub=L)
			os = []
			for i in range(n):
				oi = mdl.addVar(name="o%d" % i, lb=-L, ub=L)
				mdl.addConstr(oi == obj_terms[i])
				os += [oi]
			mdl.addConstr(o == grb.max_(os))
			obj = o# + .01*grb.quicksum(obj_terms)
		else:
			obj = grb.quicksum(obj_terms)

		obj_reg_u0, obj_reg_l0, obj_reg_u1, obj_reg_l1 = 0, 0, 0, 0

		for k in range(d1):
			obj_reg_u1 += bU1[k] * bU1[k]
			obj_reg_l1 += bL1[k] * bL1[k]

		for k in range(d0):
			obj_reg_u0 += bU0[k] * bU0[k]
			obj_reg_l0 += bL0[k] * bL0[k]

		obj_reg = self.alphau1 * obj_reg_u1 + self.alphal1 * obj_reg_l1 + \
			self.alphau0 * obj_reg_u0 + self.alphal0 * obj_reg_l0

		obj = obj + obj_reg

		mdl.addConstr((slackU0 <= self.lamdau0))
		mdl.addConstr((slackL0 <= self.lamdal0))

		mdl.addConstr((slackU1 <= self.lamdau1))
		mdl.addConstr((slackL1 <= self.lamdal1))

		mdl.setObjective(obj)
		mdl.optimize()

		self.bu0 = np.array([bU0[j].x for j in range(d0 + 1)])
		self.bl0 = np.array([bL0[j].x for j in range(d0 + 1)])

		self.bu1 = np.array([bU1[j].x for j in range(d1 + 1)])
		self.bl1 = np.array([bL1[j].x for j in range(d1 + 1)])

		return self

	def predict(self, x):
		""" Predicts lower and upper bounds at a point x """
		if self.standardize:
			x0 = self.x0scaler(x)
			x1 = self.x1scaler(x)
		else:
			x0 = x.copy()
			x1 = x.copy()

		if (self.kernel == 'rbf') or (self.kernel == 'rbf_approx'):
			x0 = self.kernel_fit(x0, tg = 0)
			x1 = self.kernel_fit(x1, tg = 1)

		else:
			x0 = self.kernel_fit(x0)
			x1 = self.kernel_fit(x1)

		yhu0 = np.dot(x0, self.bu0[:-1]) + self.bu0[-1] + self.gammau0
		yhl0 = np.dot(x0, self.bl0[:-1]) + self.bl0[-1] - self.gammal0

		yhu1 = np.dot(x1, self.bu1[:-1]) + self.bu1[-1] + self.gammau1
		yhl1 = np.dot(x1, self.bl1[:-1]) + self.bl1[-1] - self.gammal1

		return yhl0, yhu0, yhl1, yhu1

	def set_params(self, gammau0=None, gammal0=None,
						gammau1=None, gammal1=None):
		""" Sets the parameters of the model """

		if gammal1 is not None:
			self.gammal1 = gammal1
		if gammau1 is not None:
			self.gammau1 = gammau1

		if gammal0 is not None:
			self.gammal0 = gammal0
		if gammau1 is not None:
			self.gammau0 = gammau0
