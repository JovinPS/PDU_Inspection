from IPython import display
display.clear_output()

import cv2,time
from PIL import Image
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
# from IPython.display import display, Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
    
def get_seq(dict, c1, c2, c3):
    result = []
    for key, value_list in dict.items():
        if key in [c1, c2, c3]:
            for sublist in value_list:
                result.append([sublist[0][0], key])
    result.sort()
    seq=[]
    for sublist in result:
        seq.append(sublist[1])
    new=[]
    new.append(seq)
    new.append(result)
    return new

def change_bb(x,arr):
    angle_co_ordin_xyxy=x
    corner1 =angle_co_ordin_xyxy
    for i in range(len(corner1)):
        for j in range(2):
            corner1[i][j]=round(corner1[i][j])
    for i in range(len(corner1) - 1):
        start_point = tuple(corner1[i])
        end_point = tuple(corner1[i + 1])
        cv2.line(arr, start_point, end_point, (0,0,255),12)
    start_point_l = tuple(corner1[-1])
    end_point_l = tuple(corner1[0])
    cv2.line(arr, start_point_l, end_point_l, (0,0,255),12)

def cnt(data,cls):
    c = len(data.get(cls, []))
    return c

def extract_reading(roi):
    # ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    # im4=Image.fromarray(roi[..., ::-1])
    # im4.show()
    enhanced_roi = enhance_text(roi)
    zoomed_roi = cv2.resize(enhanced_roi, (0, 0), fx=2, fy=2)
    ocr_result = ocr.ocr(zoomed_roi, cls=True)
    if ocr_result and ocr_result[0]:
        # print("res",ocr_result)
        for line in ocr_result:
            # print("line",line)
            for word in line:
                # print("word",word)
                text, confidence = word[1]
                # print("text",text,"\nconf",confidence)
                if "175" in text:
                  return "175"
                elif "250" in text:
                  return "250"
                else:
                  continue
            # print(reading)
            return None

def enhance_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

def process_key(data):
    x=[]
    y=[]
    for key in {7,8}:
        if key in data:
                for sublist in data[key]:
                    if sublist:
                        x.append(sublist[0])
    return x

def findlist(data, point):
    for key,lists in data.items():
        for sublist in lists:
            if point in sublist:
                #print(sublist)
                return key,sublist
    return None, None

def get_status(model,img):
    model.to('cuda')
    # img=cv2.imread(imag)
    results1=model.predict(img,save=False,conf=0.5,device='cuda') 
    for r in results1:
        im_array=r.plot()
    obb1=results1[0].obb
    # im4 = Image.fromarray(im_array[..., ::-1])
    # im4.show()
    result_dict = {}
    result_dict1={}
    for class_index, bbox_coords in zip(obb1.cls, obb1.xyxyxyxy):
        result_dict[class_index] = bbox_coords
    for class_index , bbox_coords in zip(obb1.cls,obb1.xyxy):
        result_dict1[class_index] = bbox_coords
    new_dict = {}
    new_dict1= {}
    for key, value in result_dict.items():
        class_index = int(key.item()) 
        bbox_coordinates = value.tolist()
        if class_index not in new_dict:
            new_dict[class_index] = [bbox_coordinates]
        else:
            new_dict[class_index].append(bbox_coordinates)
    for key, value in result_dict1.items():
        class_index = int(key.item()) 
        bbox_coordinates = value.tolist()
        if class_index not in new_dict1:
            new_dict1[class_index] = [bbox_coordinates]
        else:
            new_dict1[class_index].append(bbox_coordinates)
    

    # start2 = time.time()
    req_fuse=[10,10,11,11]
    seq_fuse=get_seq(new_dict,9,10,11)
    cnt_fuse=cnt(new_dict,10)+cnt(new_dict,11)+cnt(new_dict,9)
    if(cnt_fuse==4):
        if(req_fuse==seq_fuse[0]):
            status_smallfuse="OK"
        else:
            status_smallfuse="NG"
            for i in range(4):
                if(seq_fuse[0][i]==9 or seq_fuse[0][i]==req_fuse[i]):
                    continue
                else :
                    if seq_fuse[0][i] in new_dict:
                        for j in new_dict[seq_fuse[0][i]]:
                            if j[0][0]==seq_fuse[1][i][0]:
                                change_bb(j,im_array)
                            else:
                                continue
                    else:
                        continue
    else:
        status_smallfuse="NG"
    print("smallfuse: ",status_smallfuse)

    req_hst=[6,6,5,5]
    seq_hst=get_seq(new_dict,4,5,6)
    cnt_hst=cnt(new_dict,6)+cnt(new_dict,5)+cnt(new_dict,4)
    if(cnt_hst==4):
        if(req_hst==seq_hst[0]):
            status_hst="OK"
        else:
            status_hst="NG"
            for i in range(4):
                if(seq_hst[0][i]==4 or seq_hst[0][i]==req_hst[i]):
                    continue
                else :
                    if seq_hst[0][i] in new_dict:
                        for j in new_dict[seq_hst[0][i]]:
                            if j[0][0]==seq_hst[1][i][0]:
                                change_bb(j,im_array)
                            else:
                                continue
                    else:
                        continue
    else:
        status_hst="NG"
    print("hst: ",status_hst)

    cnt_lpres=cnt(new_dict,8)
    cnt_labs=cnt(new_dict,7)
    if(cnt_lpres==2 and cnt_labs==0):
        status_levers="OK"
    elif(cnt_lpres+cnt_labs==2 or cnt_labs+cnt_lpres==1 or cnt_labs+cnt_lpres==0):
        status_levers="NG"
    else:
        status_levers="NG"
        xsort=process_key(new_dict)
        xsort.sort()
        x1,x2,x3,x4=xsort[0][0],xsort[1][0],xsort[-1][0],xsort[-2][0]
        lmin=(5*x1-x3)/4
        lmax=(3*x1+x3)/4
        rmin=(3*x3+x1)/4
        rmax=(5*x3-x1)/4
        if (lmin <= xsort[0][0] <= lmax):
            if(lmin <= xsort[1][0] <= lmax):
                if(xsort[0][1]<xsort[1][1]):
                    j=findlist(new_dict,xsort[1])[1]
                    change_bb(j,im_array)
                else:
                    j=findlist(new_dict,xsort[0])[1]
                    change_bb(j,im_array)
        if(rmin <= xsort[-1][0] <= rmax):
            if(rmin <= xsort[-2][0] <= rmax):
                if(xsort[-1][1]<xsort[-2][1]):
                    j=findlist(new_dict,xsort[-1])[1]
                    change_bb(j,im_array)
                else:
                    j=findlist(new_dict,xsort[-2])[1]
                    change_bb(j,im_array)
    print("levers: ",status_levers)

    cnt_sealpres=cnt(new_dict,3)
    if(cnt_sealpres==4):
        status_seal="OK"
    else:
        status_seal="NG"
    print("seal: ",status_seal)

    # start=time.time()
    stat=0 
    count = cnt(new_dict1,1)
    if(count==0):
        pass
    elif(count==1):
        for result in new_dict1[1]:
            x1,y1,x2,y2=map(int,result)
            x3=x2
            y3=y1
        if abs(x2-x1)>abs(y2-y1):
            x3 = int(x2 - 0.7*abs(x1-x2))
            roi=img[y3:y2,x1:x3]
            roi=cv2.rotate(roi,cv2.ROTATE_90_COUNTERCLOCKWISE)
            if(extract_reading(roi)=="250"):
                change_bb(new_dict[1][0],im_array)
                change_bb(new_dict[0][0],im_array)
            elif(extract_reading(roi)=="175"):
                change_bb(new_dict[0][0],im_array)
            else:
                x1=int(x1+0.7*abs(x1-x2))
                roi=img[y1:y2,x1:x2]
                roi=cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
                if(extract_reading(roi)=="250"):
                    change_bb(new_dict[1][0],im_array)
                    change_bb(new_dict[0][0],im_array)
                elif(extract_reading(roi)=="175"):
                    change_bb(new_dict[0][0],im_array)
        else:
            y3 = int(y1+0.7*abs(y1-y2))
            roi=img[y3:y2,x1:x3]
            if(extract_reading(roi)=="250"):
                change_bb(new_dict[0][0],im_array)
            elif(extract_reading(roi)=="175"):
                change_bb(new_dict[0][0],im_array)
                change_bb(new_dict[1][0],im_array)
            else:
                y2 = int(y2-0.7*abs(y1-y2))
                roi=img[y1:y2,x1:x2]
                roi = cv2.rotate(roi, cv2.ROTATE_180)
                if(extract_reading(roi)=="250"):
                    change_bb(new_dict[0][0],im_array)
                elif(extract_reading(roi)=="175"):
                    change_bb(new_dict[0][0],im_array)
                    change_bb(new_dict[1][0],im_array)
    elif(count==2):
        for result in new_dict1[1]:
            x1,y1,x2,y2=map(int,result)
            x3=x2
            y3=y1
            if abs(x2-x1)<abs(y2-y1):
                y3 = int(y1+0.7*abs(y1-y2))
                roi=im_array[y3:y2,x1:x3]
                if(extract_reading(roi)=="250"):
                    stat=1
                elif(extract_reading(roi)=="175"):
                    for result in new_dict[1]:
                        change_bb(result,im_array)
                else:
                    y2 = int(y2-0.7*abs(y1-y2))
                    roi=im_array[y1:y2,x1:x2]
                    roi = cv2.rotate(roi, cv2.ROTATE_180)
                    if(extract_reading(roi)=="250"):
                        stat=1
                    elif(extract_reading(roi)=="175"):
                        for result in new_dict[1]:
                            change_bb(result,im_array)
                    else:
                        stat=3
                        for result in new_dict[1]:
                            change_bb(result,im_array)

    if(stat==0):
      status_bigfuse="NG"
    elif(stat==1):
      status_bigfuse="OK"   
    else:
        status_bigfuse="-Values Not Clear-"
    print("bigfuse: ",status_bigfuse)
    # print(time.time()-start)

    im_array=cv2.resize(im_array,(640,640))
    final_image = Image.fromarray(im_array[..., ::-1])
    
    # final_image.show()
    return final_image,{'PARTS': 'STATUS','BIG FUSE': status_bigfuse , 'SMALL FUSE': status_smallfuse,'HST ': status_hst,'SEAL': status_seal, 'LEVER': status_levers}


# model=YOLO(r'D:\31jan2023-Main\ASSIGNMENTS\ultralytics\best65_ver10.pt')
# import cv2
# img_path=input("Enter path: ")
# print(get_status(model,img_path))