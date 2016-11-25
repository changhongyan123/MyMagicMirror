# -*- coding: UTF-8 -*-
import BaseHTTPServer
import json
from ocr_casia import MyModel
import numpy as np
import random
import pandas as pd
from os   import system
from photo import PhotoProcessing
import time


#服务器端配置
HOST_NAME = 'localhost'
PORT_NUMBER = 9000
#这个值是通过运行神经网络设计脚本得到的最优值

# -*- coding: UTF-8 -*-
#加载数据的说明文件 

# labels 1~6对应六种情感
# 1   angry
# 2   fear
# 3   happy
# 4   neutral
# 5   sad
# 6   surprise

import numpy as np

#data_matrix = pd.read_csv('emo_fea_unscaled.csv',delimiter=',').values
#data_labels = pd.read_csv('emo_tag.csv').values
data_labels =  np.loadtxt('emo_tag.csv',dtype=str)
data_matrix = np.loadtxt('emo_fea_unscaled.csv',delimiter=',')
#feature = pd.read_csv("feature.csv")
#data = feature.values
#random.shuffle(data)
#data_matrix = data[:,1:]
#data_labels = data[:,0]


# 数据集一共5000个数据，train_indice存储用来训练的数据的序号
train_indice = range(252)
# 打乱训练顺序
random.shuffle(train_indice)

nn = MyModel(data_matrix, data_labels)
photo_model = PhotoProcessing()

def WavToCsv(filename):
    system('cd /home/chang/opensmile-2.3.0 && ./SMILExtract -C config/IS09\_emotion.conf -I /home/chang/temp/'+filename+'.wav  -O /home/chang/data/'+filename+'.csv')
    filepath='/home/chang/data/'+filename+'.csv'
    data = pd.read_csv(filepath,skiprows=[i for i in xrange(390)])
    data = data.replace("?","0")
    test_data=data.columns.values[1:]
    test_data = test_data[:-1]
    for i in xrange(len(test_data)):
        try:
            test_data[i] = float(test_data[i])
        except:
            test_data[i] = 0
    return test_data

def make_face(filepath,emotion):
    if emotion == 'anger':
        anotherpath = '/home/chang/MagicMirror/photo/anger.jpg'
    elif emotion == 'contempt':
        anotherpath = '/home/chang/MagicMirror/photo/contempt.jpg'
    elif emotion == 'disgust':
        anotherpath = '/home/chang/MagicMirror/photo/disgust.jpg'
    elif emotion == 'fear':
        anotherpath = '/home/chang/MagicMirror/photo/fear.jpg'
    elif emotion == 'happiness':
        anotherpath = '/home/chang/MagicMirror/photo/happiness.jpg'
    elif emotion == 'neutral':
        anotherpath = '/home/chang/MagicMirror/photo/neutral.jpg'
    elif emotion == 'sadness':
        anotherpath = '/home/chang/MagicMirror/photo/sadness.jpg'
    elif emotion == 'surprise':
        anotherpath = '/home/chang/MagicMirror/photo/surprise.jpg'
    else:
        anotherpath = 'home/chang/MagicMirror/photo/test.jpg'
    system('python faceswap.py '+filepath+' '+anotherpath+'')
    



class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """处理接收到的POST请求"""
    def do_POST(self):
        print "deal post"
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)
        time.sleep(5)
        # 如果是训练请求，训练然后保存训练完的神经网络
        # 如果是预测请求，返回预测值
        if payload.get('predict'):
            filename = payload['wav']
            #print filename
            print 'deal_wav'
            response = {"type":"test", "result":str(nn.predict(WavToCsv(filename)))}
            print (nn.predict(WavToCsv(filename)))
            #print nn.predict(WavToCsv(filename)
        elif payload.get('photo_predict'):
            photoname = payload['photo']
            print photoname
            print photo_model.showphoto(photoname)
            emotion = str( photo_model.showphoto(photoname))
            pathToFileInDisk = '/home/chang/temp/'
            path = pathToFileInDisk + photoname
            make_face(path,emotion)
            response = {"type":"emotion", "result":str( photo_model.showphoto(photoname))}
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response_code:
            self.wfile.write(json.dumps(response))
        return

if __name__ == '__main__':
    print "step1"
    server_class = BaseHTTPServer.HTTPServer
    print "step2"
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)
    print "step3"
    try:
        #启动服务器
        httpd.serve_forever()
        print "start server"
    except KeyboardInterrupt:
        pass
    else:
        print "Unexpected server exception occurred."
    finally:
        print "stoppred"
        httpd.server_close()

