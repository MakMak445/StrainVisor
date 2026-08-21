import cv2
import numpy as np

print("Attempting to open a window...")
img = np.zeros((200, 200, 3), dtype=np.uint8)

cv2.imshow("GUI Test - Press any key to close", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Success!")