##
## 1.导入必要的库
##
import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW, get_cosine_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

bert_path = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(bert_path)  # 初始化分词器


##
## 2.预处理数据集
##
def tokenizer_encode(df,
					 maxlen,
					 text_col,
					 y_col):
	input_ids, input_masks = [], []  # input char ids, attention mask
	Big5_O, Big5_C, Big5_E, Big5_A, Big5_N = [], [], [], [], []

	for text in tqdm(df[text_col]):
		encode_dict = tokenizer.encode_plus(text=text,
											max_length=maxlen,
											padding='max_length',
											truncation=True)

		input_ids.append(encode_dict['input_ids'])
		input_masks.append(encode_dict['attention_mask'])

	for y in tqdm(df[y_col[0]]):
		Big5_O.append((y))
	for y in tqdm(df[y_col[1]]):
		Big5_C.append((y))			
	for y in tqdm(df[y_col[2]]):
		Big5_E.append((y))			
	for y in tqdm(df[y_col[3]]):
		Big5_A.append((y))
	for y in tqdm(df[y_col[4]]):
		Big5_N.append((y))

	input_ids, input_masks = np.array(input_ids), np.array(input_masks)
	Big5_O = np.array(Big5_O)
	Big5_C = np.array(Big5_C)
	Big5_E = np.array(Big5_E)
	Big5_A = np.array(Big5_A)
	Big5_N = np.array(Big5_N)

	return input_ids, input_masks, Big5_O, Big5_C, Big5_E, Big5_A, Big5_N


##
## 3.加载到pytorch的DataLoader
##
def data_Loader(input_ids_train, input_masks_train, y_train, input_ids_valid, input_masks_valid, y_valid, input_ids_test, input_masks_test, trian_BATCH_SIZE, test_BATCH_SIZE):
	# 训练集
	train_data = TensorDataset(torch.LongTensor(input_ids_train), 
							 torch.LongTensor(input_masks_train), 
							 torch.LongTensor(y_train))
	train_sampler = RandomSampler(train_data)  
	train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=trian_BATCH_SIZE, num_workers=0)

	# 验证集
	valid_data = TensorDataset(torch.LongTensor(input_ids_valid), 
							torch.LongTensor(input_masks_valid),
							torch.LongTensor(y_valid))
	valid_sampler = SequentialSampler(valid_data)
	valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=trian_BATCH_SIZE, num_workers=0)
	
	# 测试集（没有标签）
	test_data = TensorDataset(torch.LongTensor(input_ids_test), 
							torch.LongTensor(input_masks_test))
	test_sampler = SequentialSampler(test_data)
	test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_BATCH_SIZE, num_workers=0)

	return train_loader, valid_loader, test_loader


##
## 4.定义Roberta模型
## 
class Roberta_Model(nn.Module):
	def __init__(self, bert_path, classes=6):
		super(Roberta_Model, self).__init__()
		self.config = RobertaConfig.from_pretrained(bert_path)  # 导入模型超参数
		self.bert = RobertaModel.from_pretrained(bert_path)     # 加载预训练模型权重
		self.fc = nn.Linear(self.config.hidden_size,classes)
		
	def forward(self, input_ids, attention_mask=None):
		outputs = self.bert(input_ids, attention_mask)
		out_pool = outputs[1]   # 池化后的输出 [bs, config.hidden_size]
		logit = self.fc(out_pool)

		return logit


##
## 5.输出模型参数量
##
def get_parameter_number(model):
	total_num = sum(p.numel() for p in model.parameters())
	trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)


##
## 6.定义训练函数、验证和测试函数
## 
def evaluate(model, data_loader, device):
	# 评估模型性能，在验证集上
	model.eval()
	val_true, val_pred = [], []
	with torch.no_grad():
		for idx, (ids, att, y) in (enumerate(data_loader)):
			y_pred = model(ids.to(device), att.to(device))
			y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
			# y_pred = y_pred.detach().squeeze().cpu().numpy().tolist()
			val_pred.extend(y_pred)
			val_true.extend(y.cpu().numpy().tolist())

	# print(val_pred)
	# print(type(val_pred))
	# print(val_true)
	# print(type(val_true))

	val_pred = torch.tensor(val_pred, dtype=torch.float32)
	val_true = torch.tensor(val_true, dtype=torch.float32)

	loss = nn.CrossEntropyLoss()
	output = loss(val_pred, val_true) 

	return output # 返回CELoss


def predict(model, data_loader, device):
	# 测试集没有标签，需要预测提交
	model.eval()
	val_pred = []
	with torch.no_grad():
		for idx, (ids, att) in tqdm(enumerate(data_loader)):
			y_pred = model(ids.to(device), att.to(device))
			y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
			# y_pred = y_pred.detach().squeeze().cpu().numpy().tolist()
			val_pred.extend(y_pred)
  
	return val_pred


def train(model, train_loader, optimizer, scheduler, device, epoch, pth_save_path):
	# 训练
	best_loss = float('inf')
	criterion = nn.CrossEntropyLoss()
	
	for i in range(epoch):
		"""训练模型"""
		start = time.time()
		model.train()
		train_loss_sum = 0.0

		for idx, (ids, att, y) in enumerate(train_loader): 
			# print('y', y.shape)
			# y = y.view(-1, 1)
			# y = y.to(torch.float32)
			# print('y', y.shape)
			ids, att, y = ids.to(device), att.to(device), y.to(device)  
			y_pred = model(ids, att)
			# print(y_pred.shape)

			loss = criterion(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()   # 学习率变化
			# print(loss)

			train_loss_sum += loss.item()
			if (idx + 1) % (len(train_loader)//1) == 0:    # 只打印一次结果
				print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
						  i+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start))
				# print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

		torch.save(model.state_dict(), pth_save_path) 
		
		print("time costed = {}s \n".format(round(time.time() - start, 5)))


##
## 7. 定义整体模型
##
def RoBERTa_A(
	input_ids_train, 
	input_masks_train, 
	y_train,
	input_ids_valid, 
	input_masks_valid, 
	y_valid, 
	input_ids_test, 
	input_masks_test, 
	param_dict={'trian_BATCH_SIZE': 24, 'test_BATCH_SIZE': 64, 'lr': 2e-5, 'weight_decay': 1e-4, 'epoch': 10, 'pth_save_path': 'best_roberta_model_O.pth'}):
	
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, valid_loader, test_loader = data_Loader(input_ids_train, input_masks_train, y_train, input_ids_valid, input_masks_valid, y_valid, input_ids_test, input_masks_test, param_dict['trian_BATCH_SIZE'], param_dict['test_BATCH_SIZE'])
	model = Roberta_Model(bert_path).to(DEVICE)
	optimizer = AdamW(model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['weight_decay'])
	scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=param_dict['epoch']*len(train_loader))
	
	# 训练
	train(model, train_loader, optimizer, scheduler, DEVICE, param_dict['epoch'], param_dict['pth_save_path'])
	# 加载权重对测试集测试
	model.load_state_dict(torch.load(param_dict['pth_save_path']))
	pred_test = predict(model, test_loader, DEVICE)
	
	return pred_test
