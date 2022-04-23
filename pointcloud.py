import setup_path 
import airsim

import cv2
import time
import sys
import math
import numpy as np

outputFile = "cloud.asc" 
color = (0,255,0)
rgb = "%d %d %d" % color

W = 256
H = 144
fl = W/2
B = 20

projectionMatrix = np.array([[1, 0, 0, -W/2],
                              [0, 1, 0, -H/2],
                              [0, 0, 0, fl],
                              [0, 0, -1/B, 0]])
'''
projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                              [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                              [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                              [0.000000000, 0.000000000, -10.0000000, 0.000000000]])
'''

def savePointCloud(image, fileName):
   f = open(fileName, "w")
   for x in range(image.shape[0]):
     for y in range(image.shape[1]):
        pt = image[x,y]
        if (math.isinf(pt[0]) or math.isnan(pt[0])):
          # skip it
          None
        else: 
          f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, rgb))
   f.close()

for arg in sys.argv[1:]:
  cloud.txt = arg

client = airsim.MultirotorClient()

while True:
    rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
    png = cv2.imdecode(np.frombuffer(rawImage, np.uint8) , cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
    savePointCloud(Image3D, outputFile)
    print(Image3D)
    airsim.wait_key("Press any key to exit")
    sys.exit(0)

    key = cv2.waitKey(1) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        break;