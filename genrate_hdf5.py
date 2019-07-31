import h5py
import cv2
import os
import numpy as np
import math
import json

net_width = 48
net_height = 48
net_channel = 3
label_colum = 3 

'''
Read image and do preprocessing using cv2
'''
def image_process(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img,(net_width,net_height))
    img = img.astype(np.float64)
    img = img.transpose((2,0,1))
    mean= np.array([127.5, 127.5, 127.5])
    mean= mean[:,np.newaxis,np.newaxis]
    img-=mean
    img*=0.0078125
    return img

def generate_hdf5(file_path,save_path):
    data_list = open(file_path).read().split('\n')
    data_list.remove("")
    data_number = len(data_list)
    images = np.zeros([data_number,net_channel,net_height,net_width],dtype=np.float32)
    labels = np.zeros([data_number,label_colum],dtype=np.float32)
    for idx,each_data in enumerate(data_list):
        image_path,category,p_x,p_y = each_data.split()
        img = image_process(image_path)
        images[idx,:,:,:] = img
        labels[idx, 0] = float(category)
        labels[idx, 1] = float(p_x)
        labels[idx, 2] = float(p_y)
    save_hdf5(data_number,images,labels,save_path)

def save_hdf5(data_number,images,labels,save_path):
    batch_size = 128
    batchNum = int(math.ceil(1.0 * data_number/batch_size))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}

    for i in range(batchNum):
        start = i * batch_size
        end = min((i+1)*batch_size , batchNum)
        if i < batchNum - 1:
            fileName = save_path + '/train{0}.h5'.format(i)
        else:
            fileName = save_path + '/test{0}.h5'.format(i - batchNum + 1)
        
        with h5py.File(fileName , 'w') as f:
            f.create_dataset('data' , data=images[start : end], **comp_kwargs)
            f.create_dataset('label' , data=labels[start : end], **comp_kwargs)

        if i < batchNum - 1:
            train_file = os.path.join(save_path,"trainlist.txt")
            with open(train_file , 'a') as f:
                each_line = save_path + "/train{0}.h5".format(i) + '\n'
                f.write(each_line)
        else:
            test_file = os.path.join(save_path,'testlist.txt')
            with open(test_file, 'a') as f:
                each_line = save_path + '/test{0}.h5'.format(i - batchNum + 1) + '\n'
                f.write(each_line)


def read_json(json_file):
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

def generate_hdf5_json(folder_path,save_path):
    json_list = [each for each in os.listdir(folder_path) if os.path.splitext(each)[1] == ".json"]
    data_number = len(json_list)
    images = np.zeros([data_number,net_channel,net_height,net_width],dtype=np.float32)
    labels = np.zeros([data_number,label_colum],dtype=np.float32)
    for idx,each_json in enumerate(json_list):
        json_file = os.path.join(folder_path,each_json)
        image_file, conf, pos_x, pos_y = read_json(json_file)
        image_path = os.path.join(folder_path,image_file)
        img = image_process(image_path)
        images[idx,:,:,:] = img
        labels[idx, 0] = conf
        labels[idx, 1] = pos_x
        labels[idx, 2] = pos_y
    save_hdf5(data_number,images,labels,save_path)

if __name__ == "__main__":
    #generate_hdf5("train.txt","labels")
    generate_hdf5_json("/opt/jl/engine/SmookingDetection/call","/opt/jl/engine/SmookingDetection/labels")
