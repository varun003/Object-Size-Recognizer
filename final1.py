import cv2
import numpy as np
from scipy.spatial.distance import euclidean

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

img = cv2.imread('images/image/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(9,9),0)
canny = cv2.Canny(blur,50,100)
dilate = cv2.dilate(canny,None,iterations=2)
erode = cv2.erode(dilate,None,iterations=2)

ret,thresh = cv2.threshold(erode,50,255,cv2.THRESH_BINARY)

cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cnts = [x for x in cnts if cv2.contourArea(x) >3000]

biggest = cnts[0]

for cnt in cnts:
    # peri = cv2.arcLength(biggest,True)
    # approx = cv2.approxPolyDP(biggest,0.02*peri,True)
    # box = cv2.boundingRect(approx)
    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box)   # used with minAreaRect only
    box = np.int0(box)

    tl,tr,bl,br = box
    print('coordinates of box:',box)

    width_pixel  = round((euclidean(tl,tr)),3)
    height_pixel = round((euclidean(tr,br)),3)

    width_cm  = round((euclidean(tl,tr)/37.8),3)
    height_cm = round((euclidean(tr,bl)/37.8),3)

    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2),tl[1]+ int(abs(tr[1] - tl[1])/2))
    # print('mid_pt_horizontal',mid_pt_horizontal)
    mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0])/2),tl[1]+ int(abs(tr[1] - br[1])/2))
    # print('mid_pt_vertical',mid_pt_vertical)
    print('Width of box:',width_pixel,'Height of box:',height_pixel)
    print('Width of box:',width_cm,'Height of box:',height_cm)

    cv2.putText(img, "{:.1f} cm".format(width_cm), (int(mid_pt_horizontal[0] -15), int(mid_pt_horizontal[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(img, "{:.1f} cm".format(height_cm), (int(mid_pt_vertical[0]-100), int(mid_pt_vertical[1]-100)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2)

    cv2.drawContours(img,[box],-1,(0,255,0),2,lineType=cv2.LINE_AA)


show_images([img])
cv2.imwrite('images/result/1_sol.jpg',img)