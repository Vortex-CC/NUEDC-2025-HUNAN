from maix import camera, display, app, time, image, touchscreen
import os
import cv2
cam = camera.Camera(320, 240)
cam_work=camera.Camera(640, 480)
cam.skip_frames(50)
ts = touchscreen.TouchScreen()
disp = display.Display()

SLIDER_H   = 30
SLIDER_W   = 200
SLIDER_Y   = 180
SLIDER_X_R = 10
SLIDER_X_G = SLIDER_X_R + SLIDER_W + 20
SLIDER_X_B = SLIDER_X_G + SLIDER_W + 20

# [(0, 100, 6, 97, -68, -17)]

# r_val = g_val = b_val = 127
L_min_val=0
L_max_val=100
A_min_val=-128
A_max_val=127
B_min_val=-128
B_max_val=127
buttom_choose_flag=0

BUTTON_W, BUTTON_H = 80, 30
buttons = [
    {"label": "L_MIN", "state": 0, "pos": [10, 10, BUTTON_W, BUTTON_H]},
    {"label": "A_MIN", "state": 0, "pos": [10, 50, BUTTON_W, BUTTON_H]},
    {"label": "B_MIN", "state": 0, "pos": [10, 90, BUTTON_W, BUTTON_H]},
    {"label": "Binary", "state": 0, "pos": [10, 130, BUTTON_W, BUTTON_H]},
    {"label": "save", "state": 0, "pos": [220, 80, BUTTON_W, BUTTON_H]},
    {"label": "WORK", "state": 1, "pos": [260, 205, BUTTON_W, BUTTON_H]},
]

THRESH_FILE = '/color_lab.txt'
thr_lab = [0, 100, -128, 127, -128, 127] 

def load_thresh():
    global thr_lab,L_min_val,L_max_val,A_min_val,A_max_val,B_min_val,B_max_val
    if os.path.exists(THRESH_FILE):
        try:
            with open(THRESH_FILE, 'r') as f:
                vals = list(map(int, f.read().strip().split(',')))[:6]
                if len(vals) == 6:
                    thr_lab = vals
                    L_min_val=thr_lab[0]
                    L_max_val=thr_lab[1]
                    A_min_val=thr_lab[2]
                    A_max_val=thr_lab[3]
                    B_min_val=thr_lab[4]
                    B_max_val=thr_lab[5]
                    print('已加载阈值：', thr_lab)
        except Exception as e:
            print('读取失败，使用默认阈值：', e)
def save_thresh():
    global thr_lab,L_min_val,L_max_val,A_min_val,A_max_val,B_min_val,B_max_val
    try:
        with open(THRESH_FILE, 'w') as f:
            f.write(','.join(map(str, thr_lab)))
    except:
        pass



def clamp(v, mn, mx): return max(mn, min(v, mx))
def map_range(v, in_min, in_max, out_min, out_max):
    return int((v - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

def draw_slider(img, x, y, val, color, label):
    img.draw_rect(x, y, SLIDER_W, SLIDER_H, image.COLOR_BLUE, 2)
    if buttom_choose_flag in [0, 1]:   
        mn, mx = 0, 100
    else:  
        mn, mx = -128, 127
    fill_w = int(SLIDER_W * (val - mn) / (mx - mn))
    fill_w = clamp(fill_w, 0, SLIDER_W)

    img.draw_rect(x, y, fill_w, SLIDER_H, color, -1)
    knob_x = x + fill_w
    knob_y = y + SLIDER_H // 2
    img.draw_circle(knob_x, knob_y, 10, image.COLOR_WHITE, -1)
    img.draw_string(x, y - 15, f"{label}:{val}", image.COLOR_WHITE, scale=1)

def draw_buttons(img):
    for btn in buttons:
        x, y, w, h = btn["pos"]
        color = image.COLOR_GREEN if btn["state"] else image.COLOR_RED
        img.draw_rect(x, y, w, h, image.COLOR_BLUE, 2)
        img.draw_rect(x, y, w, h, color, -1)
        img.draw_string(x + 5, y + 8, btn["label"], image.COLOR_BLUE, scale=1)

def is_in_button(ts_x, ts_y, btn_pos):
    bx, by, bw, bh = btn_pos
    return bx <= ts_x <= bx + bw and by <= ts_y <= by + bh

def check_slider(ts_x, ts_y, slider_disp, val):
    sx, sy, sw, sh = slider_disp
    if buttom_choose_flag in [0, 1]:
        mn, mx = 0, 100
    else:
        mn, mx = -128, 127
    knob_x = sx + int(sw * (val - mn) / (mx - mn))
    knob_y = sy + sh // 2
    return abs(ts_x - knob_x) <= 100 and abs(ts_y - knob_y) <= 100


last_pressed = False
dragging = None
def get_disp_pos(orig_x, orig_y):
    return image.resize_map_pos(
        cam.width(), cam.height(), disp.width(), disp.height(),
        image.Fit.FIT_CONTAIN, orig_x, orig_y, SLIDER_W, SLIDER_H
    ) 
binary_state=0
work_flag=1
def check_button_state(btn):
    global buttom_choose_flag,binary_state,work_flag
    global thr_lab,L_min_val,L_max_val,A_min_val,A_max_val,B_min_val,B_max_val
    if btn==buttons[0]:
        if btn['state']==1:
            buttom_choose_flag=1
        else:
            buttom_choose_flag=0        
    elif btn==buttons[1]:
        if btn['state']==1:
            buttom_choose_flag=3
        else:
            buttom_choose_flag=2
    elif btn==buttons[2]:
        if btn['state']==1:
            buttom_choose_flag=5
        else:
            buttom_choose_flag=4
    elif btn==buttons[3]:
        if btn['state']==1:
            binary_state=1
        else:
            binary_state=0
    elif btn==buttons[4]:
        thr_lab[0]=L_min_val
        thr_lab[1]=L_max_val
        thr_lab[2]=A_min_val
        thr_lab[3]=A_max_val
        thr_lab[4]=B_min_val
        thr_lab[5]=B_max_val
        save_thresh()
    elif btn==buttons[5]:
        if btn['state']==1:
            work_flag=1
        else:
            work_flag=0

def  get_val_solider_by_choose(val):
    global L_min_val,L_max_val,A_min_val,A_max_val,B_min_val,B_max_val
    if buttom_choose_flag==0:
        L_min_val=val
    elif buttom_choose_flag==1:
        L_max_val=val
    elif buttom_choose_flag==2:
        A_min_val=val
    elif buttom_choose_flag==3:
        A_max_val=val
    elif buttom_choose_flag==4:
        B_min_val=val
    elif buttom_choose_flag==5:
        B_max_val=val
show_slider_val=127        
def change_show_val():
    global show_slider_val
    if buttom_choose_flag==0:
        show_slider_val=L_min_val
    elif buttom_choose_flag==1:
        show_slider_val=L_max_val
    elif buttom_choose_flag==2:
        show_slider_val=A_min_val
    elif buttom_choose_flag==3:
        show_slider_val=A_max_val
    elif buttom_choose_flag==4:
        show_slider_val=B_min_val
    elif buttom_choose_flag==5:
        show_slider_val=B_max_val
def show_lab_val():
    img.draw_string(130, 10, f"L_min:{L_min_val}", image.COLOR_BLUE, scale=1)
    img.draw_string(230, 10, f"L_max:{L_max_val}", image.COLOR_BLUE, scale=1)
    img.draw_string(130, 30, f"A_min:{A_min_val}", image.COLOR_BLUE, scale=1)
    img.draw_string(230, 30, f"A_max:{A_max_val}", image.COLOR_BLUE, scale=1)
    img.draw_string(130, 50, f"B_min:{B_min_val}", image.COLOR_BLUE, scale=1)
    img.draw_string(230, 50, f"B_max:{B_max_val}", image.COLOR_BLUE, scale=1)

def change_threshold(img):
    global last_pressed,dragging,thr_lab
    if binary_state==1:
        thresholds = ((L_min_val, L_max_val, A_min_val, A_max_val, B_min_val, B_max_val))
        img.binary([thresholds])
    change_show_val()
    draw_slider(img, SLIDER_X_R, SLIDER_Y, show_slider_val, image.COLOR_RED, "Val")
    draw_buttons(img)

    ts_x, ts_y, pressed = ts.read()

    disp_r = get_disp_pos(SLIDER_X_R, SLIDER_Y)

    if pressed:
        if not last_pressed: 
            for btn in buttons:
                btn_disp = image.resize_map_pos(
                    cam.width(), cam.height(), disp.width(), disp.height(),
                    image.Fit.FIT_CONTAIN, *btn["pos"]
                )
                if is_in_button(ts_x, ts_y, btn_disp):
                    btn["state"] = 1 - btn["state"]
                    btn["label"] = btn["label"].replace(
                        "MIN" if btn["state"] else "MAX",
                        "MAX" if btn["state"] else "MIN"
                    )
                    check_button_state(btn)
                    dragging = None
                    break
            else:
                if check_slider(ts_x, ts_y, disp_r, show_slider_val):
                    dragging = 1
        if dragging == 1:
            sx, sy, sw, sh = disp_r
            if sx <= ts_x <= sx + sw and sy <= ts_y <= sy + sh:
                if buttom_choose_flag in [0, 1]:
                    get_val_solider_by_choose(clamp(map_range(ts_x, sx, sx + sw, 0, 100), 0, 100))
                else:
                    get_val_solider_by_choose(clamp(map_range(ts_x, sx, sx + sw, -128, 127), -128, 127))
    else:
        dragging = None  
    img.draw_string(100,100,f"{buttom_choose_flag}", image.COLOR_WHITE, scale=1)
    show_lab_val()
    last_pressed = pressed
    thr_lab=[L_min_val,L_max_val,A_min_val,A_max_val,B_min_val,B_max_val]
    disp.show(img)
try:
    with open('/color_lab.txt', 'a'):
        pass
except:
    pass

try:
    with open(THRESH_FILE, 'a'):
        pass
except:
    pass

load_thresh()

def vision_prepare(img):
    global last_pressed,dragging,thr_lab

    for btn in buttons:
        if btn==buttons[5]:
            x, y, w, h = btn["pos"]
            color = image.COLOR_GREEN if btn["state"] else image.COLOR_RED
            img.draw_rect(x, y, w, h, image.COLOR_BLUE, 2)
            img.draw_rect(x, y, w, h, color, -1)
            img.draw_string(x + 5, y + 8, btn["label"], image.COLOR_BLUE, scale=1)

    ts_x, ts_y, pressed = ts.read()
    if pressed:
        if not last_pressed: 
            for btn in buttons:
                if btn==buttons[5]:
                    btn_disp = image.resize_map_pos(
                        cam.width(), cam.height(), disp.width(), disp.height(),
                        image.Fit.FIT_CONTAIN, *btn["pos"]
                    )
                    if is_in_button(ts_x, ts_y, btn_disp):
                        btn["state"] = 1 - btn["state"]
                        btn["label"] = btn["label"].replace(
                            "MIN" if btn["state"] else "MAX",
                            "MAX" if btn["state"] else "MIN"
                        )
                        check_button_state(btn)
                        dragging = None
                        break
    else:
        dragging = None  
    last_pressed = pressed
    pass


circle_threshold=[0, 100, 9, 30, -10, 27]
white_thresholds=[10,100,-128,127,-128,127]
max_blob=None

def version_work_cv(img):
    img_cv = image.image2cv(img, ensure_bgr=False, copy=False)
    img_gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray,50,100)
    contours,result=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 500:
                rect=cv2.boundingRect(approx)
                x,y,w,h=rect
                cv2.rectangle(img_cv,(x,y),(x+w,y+h),(0,255,255),2)
    img_show = image.cv2image(img_cv, bgr=True, copy=False)
    disp.show(img_show)
    fps = time.fps()      
    print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}") 

def find_maxblob(blobs):
    max_size=0
    for blob in blobs:
        if blob.area()>max_size:
            max_size=blob.area()
            max_blob=blob
    return max_blob
def draw_blob_info(blob):
    img.draw_rect(blob[0],blob[1],blob[2],blob[3],image.COLOR_GREEN)
    img.draw_cross(blob.cx(),blob.cy(),image.COLOR_GREEN)
def vision_work(img):
    img.binary([circle_threshold])
    img.dilate(1)

    # blobs=img.find_blobs([white_thresholds],merge=True, pixels_threshold=10,margin=10)
    # if blobs:
    #     max_blob=find_maxblob(blobs)
    #     draw_blob_info(max_blob)

    # vision_prepare(img)
    disp.show(img)
    fps = time.fps()      
    print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}") 
while not app.need_exit():
    if work_flag==0:
        img = cam.read()
    elif work_flag==1:
        img=cam_work.read()
    if work_flag==0:
        change_threshold(img)
    elif work_flag==1:
        vision_work(img)
        # version_work_cv(img)

