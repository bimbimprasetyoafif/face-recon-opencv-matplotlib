import matplotlib.pyplot as plt
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontface_default.xml')

img = cv2.imread('wajah6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, 1.3, 5)

c = 2
l = len(face)+1
fig = plt.figure()

fig.add_subplot(l,2,2)
plt.imshow(img)

def detectFace:
    for (x,y,w,h) in face:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        roi_color = img[y:y+h, x:x+w]
        
        c = c + 1
        fig.add_subplot(l,2,c)
        plt.imshow(roi_color)

        c = c + 1
        fig.add_subplot(l,2,c)
        plt.hist(roi_color.ravel(),256,[0,256])

detectFace()
plt.show()

