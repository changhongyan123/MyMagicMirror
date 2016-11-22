# -*- coding: UTF-8 -*-
import BaseHTTPServer
import json
from ocr_casia import MyModel
import numpy as np
import random
import pandas as pd
from os   import system
from photo import PhotoProcessing

#服务器端配置
HOST_NAME = 'localhost'
PORT_NUMBER = 9999
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


photo_model = PhotoProcessing()



class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """处理接收到的POST请求"""
    def do_POST(self):
        print "deal post"
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)
        
        if payload.get('photo_predict'):
            photoname = payload['photo']
            print photoname
            print photo_model.showphoto(photoname); 
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

