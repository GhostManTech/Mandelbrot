try:
	import multiprocessing as mp 
	import numpy as np 
	import time
	import matplotlib.pyplot as plt 
except ImportError as error:
	print(f"{error}")
else:
	resolution = 2000
	x = np.linspace(-1.25,1.25,resolution)
	y = np.linspace(-1.5,1.5, resolution)
	xx,yy = np.meshgrid(x,y)
	pairs = [(k,l) for k in range(resolution) for l in range(resolution)]
	numberCores = mp.cpu_count()

	def equation(rr, ii, ittMax=50):
		itt = 0 
		N = complex(rr,ii)
		C = complex(0.285, 	0.01)
		while abs(N) < 1e6 and itt < ittMax:
			N = N**2+C 
			itt +=1
		return ((3*itt)%256,  itt%256, (10*itt)%256)

	def mapto(pair):
		return equation(xx[pair],yy[pair])
	
	if __name__ == "__main__":
		# Without parallelization
		t1 = time.time()
		res = list(map(mapto,pairs))
		t2 = time.time()
		print(f"Sans parallélisation : {(t2-t1)}s")
		# With parallelization
		deathPool = mp.Pool(numberCores)
		t1 = time.time()
		with deathPool as p:
			res = p.map(mapto, pairs)
		t2 = time.time() 
		print(f"Avec parallélisation : {(t2-t1)}s")
		res = np.array(res)
		z = np.split(res, resolution)
		plt.imshow(z, interpolation='nearest')
		plt.show()