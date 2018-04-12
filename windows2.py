import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import cv2

from keras.preprocessing import image 
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#import imagenet_utils


model = ResNet50(weights='imagenet')
target_size = (224, 224)






def predict(model, img, target_size, top_n=20):
    
  img= np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  preds = model.predict(img)
  return preds


def plot_preds(image, preds):
  image = image.copy()
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  order = list(reversed(range(len(preds))))
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

if __name__=="__main__":
 
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  args = a.parse_args()


  Original_image=cv2.imread(args.image)#.astype('uint8')
  temp = Original_image[:,:,0].copy()
  Original_image[:,:,0] = Original_image[:,:,2]
  Original_image[:,:,2] = temp
  #Original_image = np.array(Original_image)

  if args.image is None:
    a.print_help()
    sys.exit(1)

  if args.image is not None:  
    #sub_image=Image.open(args.image)
    x=0
    y=0
    rect = [x, y, 224, 224]    #the rectangle that you are going to cut out
    block_preds = []
    for i in range(25):
      block_preds.append(np.zeros((1,1000)).astype('float64'))
    i = 0
    for y in range(0,448,112):
        for x in range(0,448,112):
          print(i)
          rect = [x, y, 224, 224]
          sub_image = Original_image[ rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2] ]
          #x=min(x,560)
          #if sub_image.size != target_size:
            #sub_image = sub_image.resize(target_size)
          preds = predict(model,sub_image.astype('float64'),target_size)
          block_preds[i] += preds
          block_preds[i+1] += preds
          block_preds[i+5] += preds
          block_preds[i+6] += preds
          i += 1
        i += 1
          

          #plot_preds(sub_image, preds)
          #x=min(x,560)
    #y=min(y,560)
          #x=x+112
    #y=y+112
    for i in range(25):
      block_preds[i] /= np.sum(block_preds[i])
      print(block_preds[i])
    
  #if args.image_url is not None:
   # response = requests.get(args.image_url)
  #  img = Image.open(BytesIO(response.content))
  #  plot_preds(img, preds)
