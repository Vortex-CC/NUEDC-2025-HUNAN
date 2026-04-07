import cv2
from maix import image

# 阈值定义
circle_threshold = [0, 100, 9, 30, -10, 27]
white_thresholds = [10, 100, -128, 127, -128, 127]

def process_image_cv(img):
    """使用OpenCV处理图像"""
    img_cv = image.image2cv(img, ensure_bgr=False, copy=False)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray, 50, 100)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 500:
                rect = cv2.boundingRect(approx)
                x, y, w, h = rect
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    return image.cv2image(img_cv, bgr=True, copy=False)

def process_image_maix(img):
    """使用MaixPy处理图像"""
    img.binary([circle_threshold])
    img.dilate(1)
    return img

def find_max_blob(blobs):
    """查找最大blob"""
    max_size = 0
    max_blob = None
    for blob in blobs:
        if blob.area() > max_size:
            max_size = blob.area()
            max_blob = blob
    return max_blob

def draw_blob_info(img, blob):
    """绘制blob信息"""
    img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
    img.draw_cross(blob.cx(), blob.cy(), image.COLOR_GREEN)