import cv2
image = cv2.imread('wallpaper.jpg')

resized = cv2.resize(image, (800, 600))
blurred = cv2.GaussianBlur(resized,(7,7),5)

cv2.imshow("origianl image",resized)
cv2.imshow("Blurred Image",blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()