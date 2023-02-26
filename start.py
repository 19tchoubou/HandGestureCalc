import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg16
import time

class GestureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.classifier(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GestureClassifier().to(device)

model.load_state_dict(torch.load('baseline_model.pth'))
model.eval()

x = 50
y = 60
w = 200
h = 200


def predictionImage(roi,thresh):
	img = np.zeros_like(roi,np.float32)

	#converting 1 channel threshold image to 3 channel image for our model
	img[:,:,0] = thresh
	img[:,:,1] = thresh
	img[:,:,2] = thresh
	img = img.reshape(1,200,200,3)
	img /= 255.

	return img

def imagePreprocess(frame):

	cv2.rectangle(frame,(x,y),(w+x,h+y),(0,255,0),2)
	roi = frame[y:h+y,x:w+x]

	hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
	#mask for thresholding the skin color
	mask = cv2.inRange(hsv,np.array([2,20,50]),np.array([30,255,255]))

	#reducing the noise in the image
	kernel = np.ones((5,5))
	blur = cv2.GaussianBlur(mask,(5,5),1)
	dilation = cv2.dilate(blur,kernel,iterations = 1)
	erosion = cv2.erode(dilation,kernel,iterations=1)
	ret,thresh = cv2.threshold(erosion,127,255,0)

	img = predictionImage(roi,thresh)
	

	return mask,thresh,img

def writeTextToWindow(img,text,default_x_calc,default_y_calc):
	fontscale = 1.0
	color = (0, 0, 0)
	fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
	cv2.putText(img, str(text), (default_x_calc, default_y_calc), fontface, fontscale, color)
	

	return img

predArray = [-1,-1]

#dimensions used while writing the predicted text
default_y_calc = 80	
default_x_calc = 25


predCount = 0 #for confirming the number displayed
predPrev = 0

#space for writing the predicted text
result = np.zeros((300,300,3),np.uint8)
result.fill(255) #fill result window(make it white)
cv2.putText(result,"Calculator", (25, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,0,0))

cap = cv2.VideoCapture(0)

t0 = time.time()

while (time.time() - t0) < 30:
	ret,frame = cap.read() #read frame
	
	mask,thresh,img = imagePreprocess(frame)

	

	#if we get the same prediction for 15 times, we take it as the confirmed prediction
	if predCount > 10:
		print('Prediction: ' + str(predPrev))
		
		#check whether it is the first operand or the second
		if predArray[0] == -1: 
			predArray[0] = predPrev
			string = '{} + '.format(predArray[0])
			writeTextToWindow(result,string,default_x_calc,default_y_calc)
			default_x_calc += 20

		else:
			default_x_calc += 40
			predArray[1] = predPrev
			string = '{} = {}'.format(predArray[1],np.sum(predArray,axis=0))
			writeTextToWindow(result,string,default_x_calc,default_y_calc)

			default_x_calc = 25
			default_y_calc += 30

			print("Sum: {}".format(np.sum(predArray,axis=0)))
			predArray = [-1,-1] #reset the values of the operands
		predCount = 0 #start counting again to get the next prediction

	
	img_torch = torch.from_numpy(img).permute(0, 3, 1, 2).to(device)

	with torch.no_grad():
		output = model(img_torch).detach().cpu().numpy()
		pred = np.argmax(output)
		#_, pred = torch.argmax(output)

	
	#increase predCount only if the previous prediction matches with our current prediction
	if predPrev == pred:
		predCount+=1
	else:
		predCount = 0

	predPrev = pred
	
	
	#showing the required windows
	cv2.imshow("result",result) #window for prediciton
	cv2.imshow('frame',frame) #main webcam window
	#cv2.imshow('roi',mask)
	cv2.imshow('thresh',thresh) #window to show the thresholded image that is being used for prediction


	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break