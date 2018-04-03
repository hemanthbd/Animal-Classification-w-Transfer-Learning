'''Program for HDR Imaging of exposures taken by a camera and tonemapping the HDR for display'''

import rawpy
import imageio
import cv2    
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from PIL import Image

def weight(z = [], *args): # Weight function to select particular pixels while throwing out unrequired ones
	z=(z-np.min(z))/(np.max(z)-np.min(z))	
	z=np.exp(-4*(np.power((z-0.5),2)/0.25))	
	return z

def exposure_time(k): # This function takes into account the time between exposures taken by the camera
	t=(1/2048)*np.power(2,k-1)
	return t

exposures=['exposure1.nef','exposure2.nef','exposure3.nef','exposure4.nef','exposure5.nef','exposure6.nef','exposure7.nef','exposure8.nef','exposure9.nef','exposure10.nef','exposure11.nef','exposure12.nef','exposure13.nef','exposure14.nef','exposure15.nef','exposure16.nef']
#Above is a list containing the exposures

for count,exp in enumerate(exposures,1): #Iterating through all of the exposures
	with rawpy.imread(exp) as raw:
		rgb = raw.postprocess(gamma=(1 ,1) , no_auto_bright=True , output_bps =16) # Color-balancing and converting the exposures into display-form
	rgb_down=cv2.resize(rgb,None,fx=0.1, fy=0.1, interpolation= cv2.INTER_CUBIC) # Downsampling them for faster computation
	imageio.imsave('processed_exposure'+str(count)+'.tiff',rgb_down) # Saving the processed exposures respectively

'''Algorithm for obtaining the HDR Image'''
#The Saved Processed Exposures in a list
filenames=['processed_exposure1.tiff','processed_exposure2.tiff','processed_exposure3.tiff','processed_exposure4.tiff','processed_exposure5.tiff','processed_exposure6.tiff','processed_exposure7.tiff','processed_exposure8.tiff','processed_exposure9.tiff','processed_exposure10.tiff','processed_exposure11.tiff','processed_exposure12.tiff','processed_exposure13.tiff','processed_exposure14.tiff','processed_exposure15.tiff','processed_exposure16.tiff']

bgr1 = cv2.imread(filenames[0]) # Reading-in the first image so as to initialize the weight and HDR functions in its shape
w= np.zeros(bgr1.shape) # Initilazing the weight function as zero
hdr=np.zeros(bgr1.shape) # Initilazing the hdr function as zero

for count,name in enumerate(filenames,1): #Iterating through the processed filenames
	bgr = cv2.imread(name)	
	bgr_norm = cv2.normalize(bgr, dst=None,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalizing the functions between 0 and 1
	w+=weight(bgr_norm) # Calculating the weight across every exposure and adding it every time to the final weight function
	hdr+=weight(bgr_norm)*bgr_norm/exposure_time(count) #HDR Calculation formula

hdr/=w # Normalizing the HDR by dividing by the total weight function
cv2.imwrite("hdr1.hdr", hdr) # Saving te HDR image

''' Algorithm for Tone Mapping- User Defined'''
I= (hdr/(1+hdr)).astype('float32') # Tone-Mapping to obtain the bright-right hand side of the image and casting into float32

cv2.imwrite("HDR_tone.jpg", I*255) # Saving the Tone-mapped Image

''' Algorithm for Tone Mapping- OpenCV by Reinhard Tone-Mapping '''
tonemap_Reinhard = cv2.createTonemapReinhard(3.0, 0,0,0) # Creating the reinhard tonemapping function
Reinhard = tonemap_Reinhard.process(hdr.astype('float32')) # Storing the result in 'Reinhard'
cv2.imwrite("Reinhard.jpg", Reinhard*255) # Saving the result

plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(Reinhard, cv2.COLOR_BGR2RGB))
plt.show()


