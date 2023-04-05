#!usr/bin/env python
#encoding:utf-8
from __future__ import division


'''
使用他们的数据集测试
'''
from model import *
from data import origin_data,label_data,xiaobo
import random


annotation = ''

input_seq_len = 10
output_seq_len = 1
dataNum = 7
output_dataNum = 1
n_in_features = dataNum
n_out_features = output_dataNum
lr=1e-3
lr_decayed = 0.5
decay=1e-5 
data_file = '../pollution_pm2.5-1.csv'
headName = ["pm25","pm10","so2","no2","co","o3"]
maxR2Score = -999 
dir_name = './model'
dir_name2 = './csv'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)
if not os.path.exists(dir_name2):
    os.makedirs(dir_name2)

parser=argparse.ArgumentParser()
parser.add_argument("model",help="Model name",type=str)
model_name=parser.parse_args().model
csvfile = open(dir_name2+'/'+model_name+'.csv','a',newline='')
header = ["index","annotation","Layers","Batch_size","Epochs","pm25_EVS","pm25_MAE","pm25_MSE","pm25_R2","100_EVS","100_MAE","100_MSE","100_R2","R2_best"]
writer = csv.writer(csvfile)
writer.writerow(header)
csvfile.close()
 
 

label = 1
if model_name == 'lstm' :model = create_lstm() 
elif model_name == 'bilstm' :model = create_bilstm() 
elif model_name == 'gru' :model = create_gru() 
elif model_name == 's2s' :model = create_s2s() 
elif model_name == 'cnn' :model = create_cnn() 
elif model_name == 'cnnlstm' :model = create_cnnlstm() 
elif model_name == 'attention' :model = create_attention() 
elif model_name == 'deattention' :model = create_deattention() 
elif model_name == 'attenlstm' :model = create_attenlstm() 
elif model_name == 'attenlstm2' :model = create_attenlstm2() 
elif model_name == 'attenlstm3' :model = create_attenlstm3() 
elif model_name == 'darnn':
    label = 2
    m = n_h = n_s = 20  #length of hidden state m
    p = n_hde0 = n_sde0 = 30  #p
    model = create_darnn()
elif model_name == 'multi':
    label = 2
    m = n_h = n_s = 32  #length of hidden state m
    p = n_hde0 = n_sde0 = 32  #p
    model = create_multi()
elif model_name == 'myModel':
    label = 3
    model = create_myModel() 
batch_size = 128
maxR2Score = -999
times = 100
maxEpochs = 400

def scheduler(epoch):
    # if epoch % 100 == 0 and epoch != 0:
    if epoch % (maxEpochs/20) == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr) 
        K.set_value(model.optimizer.lr, lr * lr_decayed)
        print("lr changed to {}".format(lr * lr_decayed))
    return K.get_value(model.optimizer.lr)

if label == 1:
    input_seq,inputSeqTest,output_seq,outputSeqTest,X_train_max,X_train_min,y_train_max,y_train_min = origin_data(data_file)
    for i in range(times):
        lr =  random.choice([0.001, 0.0001])
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay,amsgrad=False)
        model.compile(opt,loss='mean_squared_error')
        index = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M')
        decoder_input_data = np.zeros(output_seq.shape)
        reduce_lr = LearningRateScheduler(scheduler)
        history = model.fit([input_seq,decoder_input_data], output_seq,
                          batch_size=batch_size,
                          epochs=maxEpochs,
                          validation_split=0.1, 
                          shuffle=False,
                          callbacks=[ EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=2)],
                         )
        step = len(history.history['loss'])     
        decoder_test_data = np.zeros(outputSeqTest.shape)
        y_pred_all = model.predict([inputSeqTest,decoder_test_data])
        y_true_all = outputSeqTest 
        y_pred_all = np.array(y_pred_all)
        y_true_all = np.array(y_true_all)
        y_pred_all = y_pred_all * (y_train_max - y_train_min) + y_train_min
        y_true_all = y_true_all * (y_train_max - y_train_min) + y_train_min 
        y_pred_all = y_pred_all.reshape(len(y_pred_all),-1)
        y_true_all = y_true_all.reshape(len(y_pred_all),-1) 
        row = [index,lr,'-',batch_size,step]
        tmp_list = calPerformance(y_true_all ,y_pred_all)
        score = tmp_list[-1] 
        row += tmp_list
        tmp_list = calPerformance(y_true_all[:1000] ,y_pred_all[:1000])
        row += tmp_list
        row += [maxR2Score]
        if score>maxR2Score and score>0.92:
          model.save_weights(dir_name+'/'+model_name+'_weights.h5')
          maxR2Score = score
          row += ['saving...']
          print('saving...') 
        csvfile = open(dir_name2+'/'+model_name+'.csv','a',newline='')
        writer = csv.writer(csvfile)
        writer.writerow(row)
        print('Now R2_score:'+str(score))
        csvfile.close()
        
elif label==2:
    input_X_train,input_X_test,input_Y_train,input_Y_test,label_Y_train,label_Y_test,X_train_max,X_train_min,y_train_max,y_train_min,label_train_max,label_train_min = label_data(data_file)
    for i in range(times): 
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
        model.compile(opt, loss = 'mean_squared_error') 
        index = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M')
        s0_train = h0_train = np.zeros((input_X_train.shape[0],m))
        h_de0_train = s_de0_train =np.zeros((input_X_train.shape[0],p))
        history = model.fit([input_X_train,input_Y_train,s0_train,h0_train,s_de0_train,h_de0_train],label_Y_train,
            epochs=maxEpochs,batch_size=batch_size,validation_split=0.1,
            shuffle=False,
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=2)],
        )
        step = len(history.history['loss'])     
        s0_test = h0_test = np.zeros((input_X_test.shape[0],m))
        h_de0_test = s_de0_test =np.zeros((input_X_test.shape[0],p))
        pred_Y_test = model.predict([input_X_test,input_Y_test,s0_test,h0_test,s_de0_test,h_de0_test],batch_size=input_X_test.shape[0],verbose=1)
        y_pred_all = pred_Y_test * (label_train_max - label_train_min) + label_train_min
        y_true_all = label_Y_test * (label_train_max - label_train_min) + label_train_min
        y_pred_all = y_pred_all.reshape(len(y_pred_all),-1)
        y_true_all = y_true_all.reshape(len(y_pred_all),-1) 
        row = [index,'-','-',batch_size,step]
        tmp_list = calPerformance(y_true_all ,y_pred_all)
        score = tmp_list[-1] 
        calPerformance(y_true_all[:1000] ,y_pred_all[:1000])
        row += tmp_list
        row += [maxR2Score]
        if score>maxR2Score :
            model.save_weights(dir_name+'/'+model_name+'_weights.h5')
            maxR2Score = score
            row += ['saving...']
            print('saving...') 
        csvfile = open(dir_name2+'/'+model_name+'.csv','a',newline='')
        writer = csv.writer(csvfile)
        writer.writerow(row)
        print('Now R2_score:'+str(score))
        csvfile.close()

elif label == 3:
    input_seq,output_seq,inputSeqTest,outputSeqTest,X_train_max,X_train_min,y_train_max,y_train_min = xiaobo(data_file)
    for i in range(times):
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
        model.compile(opt, loss = 'mean_squared_error') 
        index = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M')
        decoder_input_data = np.zeros(output_seq.shape)
        history = model.fit([input_seq,decoder_input_data], output_seq,
                           batch_size=batch_size,
                           epochs=maxEpochs,
                           validation_split=0.1, 
                           shuffle=False,
                           callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=2)],
                          )
        step = len(history.history['loss'])     
        decoder_test_data = np.zeros(outputSeqTest.shape)
        y_pred_all = model.predict([inputSeqTest,decoder_test_data])
        y_true_all = outputSeqTest 
        y_pred_all = np.array(y_pred_all)
        y_true_all = np.array(y_true_all)
        y_pred_all = y_pred_all * (y_train_max - y_train_min) + y_train_min
        y_true_all = y_true_all * (y_train_max - y_train_min) + y_train_min 
        y_pred_all = y_pred_all.reshape(len(y_pred_all),-1)
        y_true_all = y_true_all.reshape(len(y_pred_all),-1) 
        row = [index,'-','-',batch_size,step]
        tmp_list = calPerformance(y_true_all ,y_pred_all)
        score = tmp_list[-1] 
        row += tmp_list
        tmp_list = calPerformance(y_true_all[:1000] ,y_pred_all[:1000])
        row += tmp_list
        row += [maxR2Score]
        if score>maxR2Score :
            model.save_weights(dir_name+'/'+model_name+'_weights.h5')
            maxR2Score = score
            row += ['saving...']
            print('saving...') 
        csvfile = open(dir_name2+'/'+model_name+'.csv','a',newline='')
        writer = csv.writer(csvfile)
        writer.writerow(row)
        print('Now R2_score:'+str(score))
        csvfile.close()
        

