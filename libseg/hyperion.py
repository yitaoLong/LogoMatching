import numpy as np

class hyperpro(dict):
	def __init__(self, pc=None, db={}, planes=8):
		projection = np.random.randn(planes, 512)
		if pc is not None:
			self.projection, self.center = pc
		else:
			self.projection = projection / np.linalg.norm(projection, axis = 1)[:,np.newaxis]
			self.center = np.random.randn(512)
		self.update(db)

	def project(self, x):
		y = x # - self.center
		res = [np.dot(y, i) for i in self.projection]
		return res

	def __call__(self, x):
		res= self.project(x)
		r = 0
		for idx, i in enumerate(res):
			if i > 0:
				r |= 1<<idx
		return r

	def params(self):
		# Use to store params or make a clone
		# h = hyperpro()
		# p,c = h.params()
		# g = hyperpro(p,c)
		# f = hyperpro(g.params())
		return self.projection, self.center

	def relaxed(self, y):
		numbits = y.bit_length()
		ret = set()
		while numbits > 0:
			numbits -= 1
			yy = y^(1<<numbits)
			if yy in self:
				ret |= set(self[yy])
		return ret



class hyperion():
	def __init__(self, num_hyperpros=8, num_planes=8):
		self.pros = [hyperpro(planes=num_planes) for i in range(num_hyperpros)]

	def __call__(self, x):
		ret = set()
		for h in self.pros:
			y = h(x)
			if y in h:
				ret |= set(h[y])
		return ret

	def relaxed(self, x):
		ret = set()
		for h in self.pros:
			y = h(x)
			ret |= set(h.relaxed(y))
		return ret

	def learn(self, idx, x):
		for h in self.pros:
			y = h(x)
			if y in h:
				if idx not in h[y]:
					h[y] += [idx]
			else:
				h[y] = [idx]