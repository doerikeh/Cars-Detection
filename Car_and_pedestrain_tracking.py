from cv2 import cv2

#image
img_file = "img/Car.jpg"

img = cv2.imread(img_file)

  
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Pre-Trained Car Classifier
classifer_file = "data_training/car_detector.xml"

car_tracker = cv2.CascadeClassifier(classifer_file)

cars = car_tracker.detectMultiScale(black_n_white)

for x, y, w, h in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


cv2.imshow("Car Detector", img)
cv2.waitKey()

print("Complete")
  