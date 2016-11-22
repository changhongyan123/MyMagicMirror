# -*- coding: UTF-8 -*-
import time 
import requests
import cv2
import operator
import numpy as np


# Import library to display results
import matplotlib.pyplot as plt

# Display images within Jupyter


class PhotoProcessing:
    
    def __init__(self):      #  new added  2 初始化的矩阵必须是casia特征集合才能和加载的model匹配
        # 决定了要不要导入nn.json
        # Variables
    
        self._url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        self._key = '615e8249d36b4ff3908f778e531e0f73' #Here you have to paste your primary key
        self._maxNumRetries = 10
    def processRequest( self,json, data, headers, params ):

        """
        Helper function to process the request to Project Oxford

        Parameters:
        json: Used when processing images from its URL. See API Documentation
        data: Used when processing image read from disk. See API Documentation
        headers: Used to pass the key information and the data type request
        """

        retries = 0
        result = None

        while True:

            response = requests.request( 'post', self._url, json = json, data = data, headers = headers, params = params )

            if response.status_code == 429: 

                print( "Message: %s" % ( response.json()['error']['message'] ) )

                if retries <= self._maxNumRetries: 
                    time.sleep(1) 
                    retries += 1
                    continue
                else: 
                    print( 'Error: failed after retrying!' )
                    break

            elif response.status_code == 200 or response.status_code == 201:

                if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                    result = None 
                elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                    if 'application/json' in response.headers['content-type'].lower(): 
                        result = response.json() if response.content else None 
                    elif 'image' in response.headers['content-type'].lower(): 
                        result = response.content
            else:
                print( "Error code: %d" % ( response.status_code ) )
                print( "Message: %s" % ( response.json()['error']['message'] ) )

            break

        return result
    def renderResultOnImage(self, result, img ):
        
        """Display the obtained results onto the input image"""
        
        for currFace in result:
            faceRectangle = currFace['faceRectangle']
            currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]
            if currEmotion == 'anger':
                logo = cv2.imread('/home/chang/图片/anger.jpg')
            elif currEmotion == 'contempt':
                logo = cv2.imread('/home/chang/图片/contempt.jpg')
            elif currEmotion == 'disgust':
                logo = cv2.imread('/home/chang/图片/disgust.jpg')
            elif currEmotion == 'fear':
                logo = cv2.imread('/home/chang/图片/fear.jpg')
            elif currEmotion == 'happiness':
                logo = cv2.imread('/home/chang/图片/happiness.jpg')
            elif currEmotion == 'neutral':
                logo = cv2.imread('/home/chang/图片/neutral.jpg')
            elif currEmotion == 'sadness':
                logo = cv2.imread('/home/chang/图片/sadness.jpg')
            elif currEmotion == 'surprise':
                logo = cv2.imread('/home/chang/图片/surprise.jpg')
            width=faceRectangle['width']#171
            top=faceRectangle['top']#130
            left=faceRectangle['left']#101
            height=faceRectangle['height']#169


            logo=cv2.resize(logo,((width),(height)),interpolation=cv2.INTER_CUBIC)#rows=189 cols=191

            logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            rows, cols, channels = logo.shape#
     
            roi = img[(top):(top+rows),(left):(cols+left)]#row:188 col:190
          
            # binary & mask
            ret, mask = cv2.threshold(logo_gray, 253, 255, cv2.THRESH_BINARY)
            # dst
            dst = roi
            re_row,re_col,re_channel =  dst.shape
  

            for r in xrange(re_row):#0-188
                for c in xrange(re_col):#0-190
                    if mask[r, c] == 0:
                        dst[r, c, :] = logo[r, c, :]
            img[(top):(top+rows),(left):(cols+left)] = dst
            textToWrite = "%s" % ( currEmotion )
            #cv2.putText( img, textToWrite, (faceRectangle['left'],faceRectangle['top']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1 )
            return currEmotion
    def showphoto(self,filename):
        # Load raw image file into memory
        pathToFileInDisk = '/home/chang/temp/'
        path = pathToFileInDisk + filename
        with open( path, 'rb' ) as f:
            data = f.read()
        img = cv2.imread(path)

        headers = dict()
        headers['Ocp-Apim-Subscription-Key'] = self._key
        headers['Content-Type'] = 'application/octet-stream'

        json = None
        params = None
        changedfilepath = '/home/chang/MagicMirror/modules/default/Record/image/'+filename
        result = self.processRequest( json, data, headers, params )
        #print result
        if result is not None:
            
            currEmotion=self.renderResultOnImage( result, img )
            
            cv2.imwrite(changedfilepath, img)
            print result
            #print result
            return currEmotion
        else:
            return result

           

