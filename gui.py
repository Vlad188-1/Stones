import argparse
import cv2
from cv2 import watershed
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

# from drawer import opencv-multiplot
from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image
from Colors import colors
from bounding_box import bounding_box as bb
from pathlib import Path
from datetime import datetime
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

from tracker.bot_sort import BoTSORT
from utils2.yolov5 import inference_yolo


def get_dot_on_line(x1,y1,x2,y2):
    k = (y2-y1)/(x2-x1)
    b = y1 - x1 * k
    points = {}
    for x in range(min(x1,x2), max(x1,x2)):
        points.update({int(k*x + b): x})
    for y in range(min(y1,y2), max(y1,y2)+1):
        if y not in points.keys():
            if y -1 in points.keys():
                points.update({y:points[y-1]})
            else:
                points.update({y:points[y+1]})
    return points



def get_conveer_width(polygon):
    y_min_top = max(polygon[2][1], polygon[3][1]) +1
    line1 = get_dot_on_line(*polygon[0], *polygon[1])
    line2 = get_dot_on_line(*polygon[1], *polygon[2])
    line3 = get_dot_on_line(*polygon[5], *polygon[4]) 
    line4 = get_dot_on_line(*polygon[4], *polygon[3])
    line1.update(line2)
    line3.update(line4)
    # print([(y, line3[y] - line1[y]) for y in range(y_min_top, 1080)])
    return dict([(y, line3[y] - line1[y]) for y in range(y_min_top, 1080)])
    

def update_hist():
    # summ = class1_counter + class2_counter+ class3_counter+ class4_counter
    data = {'>250': class1_counter,
            '150-200': class2_counter,
            '100-150': class3_counter,
            '80-100': class4_counter,
    }
    fig = plt.figure()

    plt.bar(data.keys(), data.values(),color = ["blue", "orange", "green", "red"])
    plt.xlabel("Размер руды")
    plt.ylabel("Количество")
    # sns.barplot(x=data.keys(), y=data.values())
    fig.canvas.draw()
    hist = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(cv2.resize(hist, (hist.shape[1]//2, hist.shape[0]//2) ))
    imgtk = ImageTk.PhotoImage(image=img)
    hist_label.imgtk = imgtk
    hist_label.configure(image=imgtk)

from collections import deque
def update_trend(
    cl1_list = deque(maxlen=10),
    cl2_list = deque(maxlen=10),
    cl3_list = deque(maxlen=10),
    cl4_list = deque(maxlen=10)
    ):
# summ = class1_counter + class2_counter+ class3_counter+ class4_counter
    # data = {'1': class1_counter,
    #         '2': class2_counter,
    #         '3': class3_counter,
    #         '4': class4_counter,
    # }
    cl1_list.append(count_per100_frames[0])
    cl2_list.append(count_per100_frames[1])
    cl3_list.append(count_per100_frames[2])
    cl4_list.append(count_per100_frames[3])
    fig = plt.figure()

    plt.plot(cl1_list, label='1', color="blue")
    plt.plot(cl2_list, label='2', color="orange")
    plt.plot(cl3_list, label='3', color="green")
    plt.plot(cl4_list, label='4', color="red")
    plt.xlabel("Время")
    plt.ylabel("Количество")
    plt.grid()
    # sns.barplot(x=data.keys(), y=data.values())
    fig.canvas.draw()
    graf = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(cv2.resize(graf, (graf.shape[1]//2, graf.shape[0]//2) ))
    imgtk = ImageTk.PhotoImage(image=img)
    graf_label.imgtk = imgtk
    graf_label.configure(image=imgtk)
    

def update_znaytost(_znyatost = deque(maxlen=10)):
    _znyatost.append(zanyatost)
    fig = plt.figure()
    plt.plot(_znyatost, label='Занятость', color="red")
    plt.xlabel("Время")
    plt.ylabel("Занятость")
    plt.title("занятость")
    plt.grid()
    fig.canvas.draw()
    graf = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(cv2.resize(graf, (graf.shape[1]//2, graf.shape[0]//2) ))
    imgtk = ImageTk.PhotoImage(image=img)
    zanyatost_label.imgtk = imgtk
    zanyatost_label.configure(image=imgtk)

def update_maxsize(_max_size = deque(maxlen=10)):
    _max_size.append(max_size)
    fig = plt.figure()
    plt.plot(_max_size, label='max_size', color="blue")
    plt.xlabel("Время")
    plt.ylabel("Размер")
    plt.title("Максимальный размер частицы")
    plt.grid()
    fig.canvas.draw()
    graf = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(cv2.resize(graf, (graf.shape[1]//2, graf.shape[0]//2) ))
    imgtk = ImageTk.PhotoImage(image=img)
    max_label.imgtk = imgtk
    max_label.configure(image=imgtk)

def draw_watershed(img, only_sum =False):
    img = cv2.resize(img, (1280,720))
    def draw_poly(mask, pts, color):
        pts = pts.reshape((-1, 1, 2))
        color = color
        mask = cv2.drawContours(mask, [pts], -1, color=color, thickness=cv2.FILLED)

        return mask

    img_gaus = cv2.GaussianBlur(img, ksize=(19, 19), sigmaX=10, sigmaY=10)
    h, w, _ = img.shape
    mask = np.ones([h, w], dtype=np.uint8)
    pts1 = np.array([
        # [141, 719],
        # [372, 361],
        # [694, 346],
        # [900, 719]
        [227, 719],
        [405, 371],
        [661, 352],
        [799, 719]
    ])
    mask = draw_poly(mask, pts1, color=0)

    pts2 = np.array([
        [415, 719],
        [454, 374],
        [581, 369],
        [626, 712],
    ])
    mask = draw_poly(mask, pts2, color=2)
    markers = mask.astype(np.int32)

    markers = cv2.watershed(img_gaus, markers)
    if only_sum:
        return (markers == 2).sum() * 2.32
    img[:, :, 2][markers == 2] = 255

    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Test track working script')
    # proga
    parser.add_argument('--mode', default='real-time')
    parser.add_argument('--save_path', default='./video_output')
    parser.add_argument("--dots", default=False, help="show track directory")
    parser.add_argument("--set_fps", type=int, default=None, help="forcibly sets fps in output video")
    parser.add_argument("--frame_start", type=int, default=None, help="number of start frame")
    parser.add_argument("--frame_stop", type=int, default=1700, help="number of last frame")
    parser.add_argument('--preset', default="camera_configs/kamni.json", help=".json that bounds area of traffic on camera")

    # yolo
    parser.add_argument('--device', default='0')
    parser.add_argument('--yolo', help='yolo path', default='./weights/fold1.pt')
    parser.add_argument('--imgsz', help='resize for this size for detection', default=640)

    # source
    parser.add_argument('--source', help='path to video file', default='/home/vik/Downloads/Kamni.mp4')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=40, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter.py out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    # CMC
    parser.add_argument("--cmc-method", default="none", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--fast_reid_config", default="./fast_reid/configs/VERIWild/bagtricks_R50-ibn.yml", type=str, help=".yaml file")
    parser.add_argument("--fast_reid_weights", default="./ReID/veriwild_bot_R50-ibn.pth", type=str, help=".yaml file")

    args = parser.parse_args()
    args.with_reid = False
    args.ablation = False
    args.name = None
    args.mot20 = not args.fuse_score
    return args


def write_video():
    pass


def is_dot_in_bound(dot, border_polygon):
    x3, y3 = dot
    logic = True
    for i in range(len(border_polygon)):
        x1, y1 = border_polygon[i-1]
        x2, y2 = border_polygon[i]
        D = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
        logic = logic and D < 0
    return logic


def draw_traeckory(img, dots, color):

    prev_dot = None
    for dot in dots:
        img = cv2.circle(img, tuple(map(int, dot)), 3, color, thickness=-1)
        if prev_dot:
            img = cv2.line(img, tuple(map(int, prev_dot)), tuple(map(int, dot)), color, thickness = 1)
        prev_dot = dot
    return img


def draw_polygon(img, dots):
    if len(dots) < 3:
        raise Exception('Кол-во точек в .json файле меньше 3, не получается построить замкнутую фигуру')
    # print(np.array(dots, dtype=np.int32))
    image = cv2.polylines(img, np.array([dots], dtype=np.int32), True, (0, 0, 255), 2)
    return image


def parse_preset(preset_path):
    with open (preset_path, 'r') as file:
        data = json.load(file)
        return data


def remove_negative(numbers):
    result = []
    for number in numbers:
        if number < 0:
            result.append(0)
        else:
            result.append(number)
    return result


def get_size_online(t,l,b,r):
    if r > 1080: 
        r = 1079
    elif r < 523:
        r = 523
    else: 
        pass
    global max_size
    size = ((b - t) * 1600) // conveer_width[int(r)]
    
    if size > max_size:
        max_size = size

    return size


def get_size_mean(sizes):
    return int(sum(sizes) / len(sizes))


def count_sizes(tarcker):
    global dots
    global class1_counter, class2_counter, class3_counter, class4_counter
    tracks = tracker.get_tracks_to_count()
    if tracks:
        for track in tracks:
            if track.track_id in dots.keys():
                size = get_size_mean(dots[track.track_id])
                if size > 250:
                    class1_counter += 1
                    count_per100_frames[0] +=1
                    #clas1_label.configure(text = str(class1_counter))
                elif size < 250 and size > 150:
                    class2_counter += 1
                    count_per100_frames[1] +=1
                    #clas2_label.configure(text = str(class2_counter))
                elif size < 150 and size > 100:
                    class3_counter += 1
                    count_per100_frames[2] +=1
                    #clas3_label.configure(text = str(class3_counter))
                elif size < 100 and size > 80:
                    class4_counter += 1
                    count_per100_frames[3] +=1
                    #clas4_label.configure(text = str(class4_counter))
                dots.pop(track.track_id)

     

def video_stream():
    global counter
    global dots
    global color_counter
    global class1, class2, class3, class4
    global count_per100_frames, max_size

    # read and check exit condition
    ret, img = video.read()
    if not ret:
        video.release()
        if args.mode == "video":
            output.release()
        return
    counter +=1
    if args.preset:       # draw bounding border
        img = draw_polygon(img, border_polygon)

    # yolo predictions
    dets = inference_yolo(
        yolo, img, device, args.imgsz)
    

    # preparing dets that are in border_polygon
    if border_polygon and False:
        pass
    else:
        needed_dets = dets

    mask = (np.logical_or(needed_dets[:, 5] == 0, needed_dets[:, 5] == 4))
    online_targets = tracker.update(needed_dets[mask][:, :5], img)
    new_img = img.copy()
    frame_dots = []
    
    if counter % 18 ==0:
        global zanyatost
        zanyatost =draw_watershed(new_img, only_sum= True)/all_s
        update_znaytost()
        # print("Занятость", draw_watershed(new_img, only_sum= True)/all_s)
    for x in online_targets:

        t, l, b, r = remove_negative(x.tlbr)
        if not x.color:
            x.color = colors[color_counter % len(colors)]
            color_counter += 1

        if x.track_id in dots.keys():
            dots[x.track_id].append(get_size_online(t,l,b,r))

        else:
            dots[x.track_id] = [get_size_online(t,l,b,r)]
        frame_dots.extend(dots[x.track_id])

        if args.mode == "video" or args.mode == "real-time":
            if shwo_watershet.get():
                new_img = draw_watershed(new_img)
            else:    
                if get_size_mean(dots[x.track_id]) > int(w.get()):
                    bb.add(new_img, t, l, b, r, str(x.track_id) + "_" + str(get_size_mean(dots[x.track_id])), (0, 0, 255)) #get_size_mean(dots[x.track_id])(x.color[2], x.color[1], x.color[0])
                else:
                    bb.add(new_img, t, l, b, r, str(x.track_id) + "_" + str(get_size_mean(dots[x.track_id])), (255,0,0))
    
    if shwo_watershet.get():
        new_img = draw_watershed(new_img)

    if args.mode == "real-time":
        img = Image.fromarray(cv2.resize(new_img, (1280,720)))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    
   
    if counter % 25 ==0:
        update_hist()

    if counter % 50 == 0:
        update_trend()
        update_maxsize()
        max_size =0
        count_per100_frames = [0,0,0,0]

    count_sizes(tracker)
    lmain.after(1, video_stream) 


dots = {}





if __name__ == "__main__":
    args = parse_args()
    class1, class2, class3, class4 = (0, 0, 0, 0)
    counter =0
    zanyatost =0
    all_s = 534475
    #Tk
    root = Tk()
    count_per100_frames = [0,0,0,0]
    max_size = 0
    app = Frame(root)#, bg="white")
    app.grid()
    
    lmain = Label(app)
    lmain.grid()

    bot_frame = Frame(root)
    bot_frame.grid(row=1, column=0,sticky=W)
    label = Label(bot_frame, text = "Выберите порог более чем:")
    label.grid(row=0, column =0,sticky=W)
    w = Scale(bot_frame, from_=40, to=250, orient=HORIZONTAL)
    w.grid(row=0, column =1,sticky=W)

    right_frame = Frame(root)
    right_frame.grid(row=0, column =1,sticky=NW)

    # Label(right_frame, text="Класс 1 (>250): ").grid(row=0, column =0,sticky=NW) 
    # clas1_label = Label(right_frame, text = "0")
    # clas1_label.grid(row=1, column =0,sticky=N, pady=(0,10))
    class1_counter = 0

    # Label(right_frame, text="Класс 2 (150-250): ").grid(row=2, column =0,sticky=NW) 
    # clas2_label = Label(right_frame, text = "0")
    # clas2_label.grid(row=3, column =0,sticky=N, pady=(0,10))
    class2_counter = 0

    # Label(right_frame, text="Класс 3 (100-150): ").grid(row=4, column =0,sticky=NW) 
    # clas3_label = Label(right_frame, text = "0")
    # clas3_label.grid(row=5, column =0,sticky=N, pady=(0,10))
    class3_counter = 0

    # Label(right_frame, text="Класс 4 (80-100): ").grid(row=6, column =0,sticky=NW) 
    # clas4_label = Label(right_frame, text = "0")
    # clas4_label.grid(row=7, column =0,sticky=N, pady=(0,10))
    class4_counter = 0
    max_label = Label(right_frame)#, text = "0")
    max_label.grid(row=0, column =0,sticky=NW) 
    ttk.Separator(right_frame, orient='horizontal').grid(row=7, column =0,sticky=N, pady=(10,10))
    shwo_watershet = IntVar()
    enabled_checkbutton = ttk.Checkbutton(right_frame, text="Отображение всей фракции", variable=shwo_watershet)
    enabled_checkbutton.grid(row=8, column =0,sticky=N, pady=(0,10))

    hist_label = Label(right_frame)
    hist_label.grid(row=9, column =0,sticky=N, pady=(0,10))
    
    graf_label = Label(right_frame)
    graf_label.grid(row=10, column =0,sticky=N, pady=(0,10))
    
    zanyatost_label  = Label(right_frame)
    zanyatost_label.grid(row=0, column =1,sticky=N, pady=(0,10))
    # yolo
    device = select_device(args.device)
    yolo_path = args.yolo
    yolo = DetectMultiBackend(yolo_path, device=device)

    # video settings
    video = cv2.VideoCapture(args.source)
    fps = video.get(cv2.CAP_PROP_FPS)
    video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # bot-sort
    tracker = BoTSORT(args, frame_rate=fps)

    if args.mode == "video":
        # output video
        DST = Path(args.save_path) / (str(datetime.now()) + ".mp4")
        encode = cv2.VideoWriter_fourcc(*'mp4v')
        print("Video will be saved in ", str(DST))

        if args.set_fps: # forcibly sets fps in output video
            output = cv2.VideoWriter(str(DST), encode, args.set_fps, video_size)
        else:
            output = cv2.VideoWriter(str(DST), encode, fps, video_size)

    # border polygon
    if args.preset:
        border_polygon = parse_preset(args.preset)["area"]
    else:
        border_polygon = None
    # preparation
    total = video.get(cv2.CAP_PROP_FRAME_COUNT)
    toc = time.time()
    color_counter = 0

    conveer_width = get_conveer_width(border_polygon)

    video_stream()
    root.mainloop()

  
   

