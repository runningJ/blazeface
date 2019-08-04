import os
import cv2
import json
from data_augmentor import Data_augmentor
import shutil

da_obj = Data_augmentor()

def flip_enhance(image_file,left_x,right_x):
    image = cv2.imread(image_file)
    flip_left_x = 1 - left_x
    flip_right_x = 1-right_x
    flip_image = cv2.flip(image,1)
    if left_x == 0.0 and right_x == 0.0:
        flip_left_x = 0.0
        flip_right_x = 0.0
    return flip_image,flip_left_x,flip_right_x


def read_json(json_file):
    file = open(json_file, "rb")
    filejson = json.load(file)
    image_file = str(filejson['image']['information']['name'])
    bbox_list = filejson['annotation']['persons']
    if len(bbox_list) == 1:
        left_x =float(bbox_list[0]['box'][0])
        right_x = float(bbox_list[0]['box'][2])
    else:
        left_x = 0.0
        right_x = 0.0
    return image_file,left_x,right_x

def write_json(json_file,write_json,image_name,left_x,right_x):
    file = open(json_file, "rb")
    filejson = json.load(file)
    filejson['image']['information']['name'] = image_name
    if left_x != 0.0 or right_x != 0.0:
        filejson['annotation']['persons'][0]['box'][0]=str(left_x)
        filejson['annotation']['persons'][0]['box'][2]=str(right_x)
    with open(write_json,'w') as newf:
        json.dump(filejson,newf)

def write_json_bright(json_file,write_json,image_name):
    file = open(json_file, "rb")
    filejson = json.load(file)
    filejson['image']['information']['name'] = image_name
    with open(write_json,'w') as newf:
        json.dump(filejson,newf)

#process one folder do flip enhance
def do_flip_enhance(folder_path,save_path):
    json_list = [each for each in os.listdir(folder_path) if os.path.splitext(each)[1] == ".json"]
    for each_json in json_list:
        json_file = os.path.join(folder_path,each_json)
        image_file,left_x, right_x = read_json(json_file)
        image_path = os.path.join(folder_path,image_file)
        flip_image,flip_left_x,flip_right_x = flip_enhance(image_path,left_x,right_x)

        new_image_file = os.path.splitext(image_file)[0]+"_flip.jpg"
        save_image_path = os.path.join(save_path,new_image_file)
        cv2.imwrite(save_image_path,flip_image)
        new_json_file = os.path.splitext(each_json)[0]+"_flip.json"
        save_json_path = os.path.join(save_path,new_json_file)
        write_json(json_file,save_json_path,new_image_file,flip_left_x,flip_right_x)

#process one folder do bright enhance
def do_bright_enhance(folder_path,save_path,factor):
    str_factor = str(factor).replace('.','-')
    json_list = [each for each in os.listdir(folder_path) if os.path.splitext(each)[1] == ".json"]
    for each_json in json_list:
        json_file = os.path.join(folder_path,each_json)
        image_file,left_x, right_x = read_json(json_file)
        image_path = os.path.join(folder_path,image_file)
        res, method = da_obj.change_brightness(image_path,factor)
        
        new_image_file = os.path.splitext(image_file)[0]+"_" +method+"_"+str_factor+".jpg"
        save_image_path = os.path.join(save_path,new_image_file)
        res.save(save_image_path)
        new_json_file = os.path.splitext(each_json)[0]+"_" +method+"_"+str_factor+".json"
        save_json_path = os.path.join(save_path,new_json_file)
        write_json_bright(json_file,save_json_path,new_image_file)

def read_json_for_test(json_file):
    file = open(json_file, "rb")
    filejson = json.load(file)
    image_file = str(filejson['image']['information']['name'])
    bbox_list = filejson['annotation']['persons']
    if len(bbox_list) == 1:
        conf = 1.0
        pos_x = (float(bbox_list[0]['box'][0]) + float(bbox_list[0]['box'][2]))/2
        pos_y = (float(bbox_list[0]['box'][1]) + float(bbox_list[0]['box'][3]))/2
    else:
        conf = 0.0
        pos_x = 0.0
        pos_y = 0.0
    return image_file,conf,pos_x,pos_y

#test the accuracy of flip enhance
def test_correct(folder_path):
    json_list = [each for each in os.listdir(folder_path) if os.path.splitext(each)[1] == ".json"]
    for each_json in json_list:
        print (each_json)
        json_file = os.path.join(folder_path,each_json)
        image_file,conf,pos_x,pos_y = read_json_for_test(json_file)
        image_path = os.path.join(folder_path,image_file)
        img = cv2.imread(image_path)
        img_h,img_w,_ = img.shape
        if pos_x != 0.0:
            real_pos_x = int(pos_x*img_w)
            real_pos_y = int(pos_y*img_h)
            cv2.circle(img,(real_pos_x,real_pos_y),2,(0,0,255),4)
            cv2.imshow("image",img)
            cv2.waitKey(0)


#generate no object label
def generate_empty_label(folder_path,empty_json):
    image_list = [image for image in os.listdir(folder_path) if os.path.splitext(image)[1] in [".jpg",".png"]]
    for image in image_list:
        suffix_name = os.path.splitext(image)[1]
        json_name = image.replace(suffix_name,".json")
        write_json_file = os.path.join(folder_path,json_name)
        write_json_bright(empty_json,write_json_file,image)


if __name__ == "__main__":
    do_flip_enhance("empty20190801","calltrain")
    #test_correct("empty20190801")
    #do_bright_enhance("transcall20190801","enhancecallbright",0.8)
    #generate_empty_label("empty20190801","empty.json")