import math
import numpy as np
import pandas as pd

class DataLoader():
	"""A class for loading and transforming data for the lstm model"""

	def __init__(self, filename, split, cols):
		# fetch different periods to get
		dataframe = pd.read_csv(filename)
		# percent
		i_split = int(len(dataframe) * split)
		# cols只有price和volume，就不能带着时间index一起吗？
		self.data_train = dataframe.get(cols).values[:i_split]
		self.data_test  = dataframe.get(cols).values[i_split:]
		# length
		self.len_train  = len(self.data_train)
		self.len_test   = len(self.data_test)
		self.len_train_windows = None

	# 尴尬，我不会直接引用__init__里的值，所以才多写了这个函数
	def get_split_data(self):
		true_data_train = self.data_train
		true_data_test = self.data_test
		return true_data_train,true_data_test


	def get_test_data(self, seq_len, normalise):
		'''
		Create x, y test data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise reduce size of the training split.
		'''
		data_windows = []
		for i in range(self.len_test - seq_len):
			data_windows.append(self.data_test[i:i+seq_len])

		#print ("data_windows"+ data_windows.shape)
		data_windows = np.array(data_windows).astype(float)
		data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

		# print ("test_data_windows"+data_windows)
		x = data_windows[:, :-1]
		#print("test_x.shape"+x.shape)
		y = data_windows[:, -1, [0]]
		#print("test_y.shape"+y.shape)
		return x,y

	def get_train_data(self, seq_len, normalise):
		'''
		Create x, y train data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise use generate_training_window() method.
		'''
		data_x = []
		data_y = []
		for i in range(self.len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise)
			data_x.append(x)
			data_y.append(y)
		# should not be tuple
		# print("train_data_x.shape"+np.array(data_x).shape)
		# print("train_data_y.shape" + np.array(data_y).shape)

		return np.array(data_x), np.array(data_y)

	def generate_train_batch(self, seq_len, batch_size, normalise):
		'''Yield a generator of training data from filename on given list of cols split for train/test'''
		i = 0
		while i < (self.len_train - seq_len):
			x_batch = []
			y_batch = []
			for b in range(batch_size):
				if i >= (self.len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't divide evenly
					yield np.array(x_batch), np.array(y_batch)
				x, y = self._next_window(i, seq_len, normalise)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
			yield np.array(x_batch), np.array(y_batch)

	def _next_window(self, i, seq_len, normalise):
		'''Generates the next data window from the given index location i'''
		window = self.data_train[i:i+seq_len]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		x = window[:-1]
		y = window[-1, [0]]

		return x, y

	def normalise_windows(self, window_data, single_window=False):
		'''Normalise window with a base value of zero'''
		normalised_data = []
		window_data = [window_data] if single_window else window_data
		for window in window_data:
			normalised_window = []
			for col_i in range(window.shape[1]):
				#normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
				normalised_col = []
				for p in window[:, col_i]:
					if window[0, col_i] != 0:
						normalised_col.append(((float(p) / float(window[0, col_i])) - 1))  # （926，2）
					else:
						normalised_col.append(0.5)
				normalised_window.append(normalised_col)
			normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
			normalised_data.append(normalised_window)
		return np.array(normalised_data)
