import sys, cv2, numpy as np

if len(sys.argv) < 2:
    print("Használat: python hsv_pick.py path/to/image.jpg")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print("Nem találom a képet:", sys.argv[1]); sys.exit(2)

win = "HSV picker (kattints a cukorkákra; ESC-kilép)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y, x].astype(np.uint8)
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0,0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        print(f"HSV @ ({x},{y})  ->  H:{h}  S:{s}  V:{v}")

cv2.setMouseCallback(win, on_mouse)

while True:
    cv2.imshow(win, img)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        break
cv2.destroyAllWindows()
