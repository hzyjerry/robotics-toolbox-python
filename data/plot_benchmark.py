import os
import numpy as np

data = {} # nq -> method -> num_pts -> {before: [], after: [], main: [], iteration: []}
# TODO: video screenshot
# TODO: robot part screenshot

def read(file):
	with open(file) as f:
		num_pts = int(file.split("_pts")[1].split("_nq")[0])
		nq = int(file.split("_nq")[1].split(".txt")[0])
		print(num_pts, nq)
		if nq not in data:
			data[nq] = {}
		method = None
		for line in f:
			if "CHOMP Mode" in line:
				method = line.split("CHOMP Mode ")[1]
				if method not in data[nq]:
					data[nq][method] = {}
				if num_pts not in data[nq][method]:
					data[nq][method][num_pts] = {"before": [], "after": [], "main": [], "iteration": []}

			if "Before: complete:" in line: # Before: complete: 7281.78 fps, 0.00014 seconds
				fps = float(line.split(" ")[2])
				sec = float(line.split(" ")[4])
				if fps > 1:
					data[nq][method][num_pts]["before"].append(1 / fps)
				else:
					data[nq][method][num_pts]["before"].append(sec)
			elif "Main: complete:" in line: # Main: complete: 4670.72 fps, 0.00021 seconds
				fps = float(line.split(" ")[2])
				sec = float(line.split(" ")[4])
				if fps > 1:
					data[nq][method][num_pts]["main"].append(1 / fps)
				else:
					data[nq][method][num_pts]["main"].append(sec)
			elif "After: complete:" in line: # After: complete: 6853.44 fps, 0.00015 seconds
				fps = float(line.split(" ")[2])
				sec = float(line.split(" ")[4])
				if fps > 1:
					data[nq][method][num_pts]["after"].append(1 / fps)
				else:
					data[nq][method][num_pts]["after"].append(sec)
			elif "iteration: complete:" in line: # Mode NONE iteration: complete: 25.99 fps, 0.03847 seconds
				fps = float(line.split(" ")[4])
				data[nq][method][num_pts]["iteration"].append(1 / fps)
		# print(data)

def plot_scaling(nq=10):
	import matplotlib
	import matplotlib.pyplot as plt
	font = {'family' : 'normal',
	        'size'   : 15}
	matplotlib.rc('font', **font)
	fig, ax = plt.subplots(figsize=(12, 8))
	for method, exp in data[nq].items():
		method_itr_mean, method_itr_std = [], []
		method_bef_mean, method_bef_std = [], []
		method_aft_mean, method_aft_std = [], []
		method_main_mean, method_main_std = [], []
		xs = sorted(list(exp.keys()))
		for num_pts in xs:
			result = exp[num_pts]
			method_itr_mean.append(np.array(result['iteration']).mean())
			method_itr_std.append(np.array(result['iteration']).std())
		# print(method_itr.mean())
		ax.plot(xs, method_itr_mean, label=f"Method: {method}", linewidth=4.0)
		# ax.errorbar(xs, method_itr_mean, yerr=method_itr_std)
	ax.set(xlabel='Number of points per robot link', ylabel='Time (s)',
	       title='Scaling plot with input points')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend()
	fig.savefig("scaling.png")
	# plt.show()


def plot_profile(nq=10):
	import matplotlib
	import matplotlib.pyplot as plt
	font = {'family' : 'normal',
	        'size'   : 15}
	matplotlib.rc('font', **font)
	fig, axes = plt.subplots(2, 2, figsize=(18, 12))
	idx = 0
	for method, exp in data[nq].items():
		ax = axes[int(idx/2)][idx % 2]
		method_itr_mean, method_itr_std = [], []
		method_bef_mean, method_bef_std = [], []
		method_aft_mean, method_aft_std = [], []
		method_main_mean, method_main_std = [], []
		xs = sorted(list(exp.keys()))
		for num_pts in xs:
			result = exp[num_pts]
			method_bef_mean.append(np.array(result['before']).mean())
			method_bef_std.append(np.array(result['before']).std())
			method_aft_mean.append(np.array(result['after']).mean())
			method_aft_std.append(np.array(result['after']).std())
			method_main_mean.append(np.array(result['main']).mean())
			method_main_std.append(np.array(result['main']).std())
		total = np.array(method_bef_mean) + np.array(method_main_mean) + np.array(method_aft_mean)
		bef = np.array(method_bef_mean) / total * 100
		main = np.array(method_main_mean) / total * 100
		aft = np.array(method_aft_mean) / total * 100
		# print(method_itr.mean())
		ax.plot([],[],color='orange', label='pre', linewidth=3)
		ax.plot([],[],color='green', label='cuda', linewidth=3)
		ax.plot([],[],color='red', label='post', linewidth=3)
		# ax.plot([],[],color='black', label='play', linewidth=3)
		ax.stackplot(xs, bef, main, aft, colors=["orange", "green", "red"])
		# ax.errorbar(xs, method_itr_mean, yerr=method_itr_std)
		ax.set_xlabel(f"Method: {method}")
		ax.set_ylabel("Percentage")
		ax.set_xscale('log')
		ax.legend()
		idx += 1
	# ax.set(xlabel='Number of points per robot link', ylabel='Time (s)',
	       # title='Scaling plot with input points')
	fig.suptitle("Profiling")
	fig.tight_layout()
	fig.savefig("profile.png")
	# plt.show()

if __name__ == "__main__":
	filepaths = []
	for file in os.listdir("data"):
		if ".txt" in file and "nq" in file:
			if "6400" in file:
				continue
			filepaths.append(os.path.join("data", file))

	for file in filepaths:
		read(file)

	plot_scaling(nq=10)
	plot_profile(nq=10)