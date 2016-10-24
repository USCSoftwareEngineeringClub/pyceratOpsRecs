import sys
import numpy as np
import cv2
"""
routine for image segmentation and 
detection of numbers and arithemic operations

opencv 3.0+

"""

def segment(im):
    """
    :param im:
        Image to detect digits and operations in

    """

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) #smooth image to reduce noise
    #adaptive thresholding for different lighting conditions
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    #################     Now finding Contours     ###################
    image,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100), np.float32)
    responses = []
    keys = [i for i in range(48,58)]

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x,y,w,h] = cv2.boundingRect(cnt)

            #Draw bounding box for it, then resize to 10x10, and store its pixel values in an array
            if  h>1:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('detecting',im)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys: #press any key to continue
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "segmentation complete"

    samples = np.float32(samples)
    responses = np.float32(responses)

    cv2.imwrite('data/seg_result.png',im)
    np.savetxt('data/generalsamples.data',samples)
    np.savetxt('data/generalresponses.data',responses)

def main():
     im = cv2.imread('data/img.png')
     segment(im)


if __name__ == "__main__":
    main()

