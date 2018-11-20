import os
import json
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from datetime import datetime,timedelta
import math

configs = json.load(open('config.json', 'r'))
tt = "LSTM_%s days_prediction"%(configs['data']['sequence_length'])

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	title =  tt
	plt.title(title)
	#Pad the list of predictions to shift it in the graph to its correct start
	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data) #label='Prediction'
		plt.legend()
	plt.savefig(tt + ".png")
	plt.show()

# Tips for debugging
# 1. 一次不要点太多断点，因为它运行到一个断点就会停住，之后的程序都不会北运行到，也就起不到debug的作用了
# 2. 不要在方程里点断点，而是在调用这个方程的地方点，不然啥debug的内容都不会出来，因为直接运行function是不会有变量信息的，只有在调用传参的时候才会

# 将变化率还原成价格
unnorm_pre = []
def predict_unnorm_data(y_test_true,prediction_len,predictions):
	# y_test_true是二维/列的，['price','volume']
	y_test_true = y_test_true[:, 0]
	# y_test_true: (301,)
	# 发现predictions的20*14,所以i也要控制在20
	# 因为y_test_true多出的部分（301/14=21.5），第21行多出的不论几个数据都要舍掉，所以要减去这一行，从而与预测数据大小匹配
	# int（），向下取整
	for i in range(int(len(y_test_true)/prediction_len)-1):
		predicted_line = []
		for j in range(prediction_len):
			p = y_test_true[i * (prediction_len)]
			p = p*(1+predictions[i][j])
			predicted_line.append(p) #14, right
		unnorm_pre.append(predicted_line)
	return unnorm_pre

# y_test_true输入的时候，应该是带有时间index的dataframe
def calc_RMSE(predicted_data, y_test_true,begin_date,end_date):
	# 统一时间index
	predicted_data = np.array(predicted_data).reshape(-1)
	predicted_data = pd.DataFrame(predicted_data)
	predicted_data.columns = ['predicted_data']
	# 取price那一列
	y_test_true = y_test_true[0]
	# 因为真实值的长度会比flatten之后的预测值长，所以要统一矩阵大小
	n_shape = len(predicted_data)
	y_test_true = y_test_true[:n_shape]
	y_test_true = pd.DataFrame(y_test_true)
	y_test_true.columns = ['y_test_true']
	# 统一index
	time = y_test_true.index.values
	# type of time is list
	predicted_data['time'] = time
	predicted_data = predicted_data.set_index('time')
	df = pd.concat([y_test_true, predicted_data], axis=1)
	# 计算RMSE
	mse = mean_squared_error(y_test_true[begin_date:end_date], predicted_data[begin_date:end_date])
	rmse = math.sqrt(mse)
	df['rmse'] = rmse
	print(df[begin_date:end_date])


def main():
	# instantiation
	data = DataLoader(
		os.path.join('data', configs['data']['filename']),
		configs['data']['train_test_split'],
		configs['data']['columns']
	)



	model = Model()
	model.build_model(configs)
	x, y = data.get_train_data(
		seq_len = configs['data']['sequence_length'],
		normalise = configs['data']['normalise']
	)

	'''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size']
	)
	'''
	# out-of memory generative training
	# 不懂为什么要自己算这个，LSTM不是自己自带batch参数吗，为什么要自己算出来一共要输多少次batch？？？

	# 会出现：
	# in data_generator_task， generator_output = next(self._generator)， StopIteration
	# in fit_generator， str(generator_output))
	# output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: None
	# 所以出错的时候，手动减少steps_per_epoch
	steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])-7
	model.train_generator(
		data_gen = data.generate_train_batch(
			seq_len = configs['data']['sequence_length'],
			batch_size = configs['training']['batch_size'],
			normalise = configs['data']['normalise']
		),
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		steps_per_epoch = steps_per_epoch
	)

	x_test, y_test = data.get_test_data(
		seq_len = configs['data']['sequence_length'],
		normalise = configs['data']['normalise']
	)

	predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
												   configs['data']['sequence_length'])
	plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

	# predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
	# predictions = model.predict_point_by_point(x_test)

	y_true_train, y_true_test = data.get_split_data()
	unnorm_data = predict_unnorm_data(y_true_test, prediction_len=configs['data']['sequence_length'],predictions=predictions)
	# 计算RMSE并输出dataframe
	begin_date = datetime(year=2018, month=9, day=18)
	end_date = begin_date + timedelta(days=(configs['data']['sequence_length']-1))
	# y_true_test：（301，2）
	y_true_test = pd.DataFrame(y_true_test)
	file = pd.read_csv(os.path.join('data', configs['data']['filename']))
	file = file['time'][len(y_true_train):]
	file = pd.Series(file)
	# 出现了无法新建列并赋值的error
	# 因为dataframe和Series都有自己的index，.values才能取到真正的值并赋给下一个变量
	y_true_test['time'] = file.values
	y_true_test = y_true_test.set_index('time')
	y_true_test.index = pd.to_datetime(y_true_test.index)
	calc_RMSE(predicted_data=unnorm_data,y_test_true=y_true_test, begin_date=begin_date,end_date=end_date)


if __name__=='__main__':
	main()



# Unsolved Problems:
# use 90 window_size, which means shift/create 90 features/columns for one feature before, and then use these new features to predict close price(y).
# take more epoches, and apply early stopping.

