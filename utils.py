__author__="xu hongtao"
#email: xxxmy@foxmail.com
#wechat: 371802059
#create:03/30/2019

from labelme import utils
import json
import cv2
import os
import numpy as np



def get_all_class_mask(json_file_path):
    '''
    returns:  
    img: 原图  
    mask: 所有类别的mask,二值图，0和255  
    img_with_label: 带掩模的图  
    lbl_value_to_names: mask的index可以从lbl_value_to_names得到class名称  
    '''
    if os.path.exists(json_file_path):
        with open(json_file_path) as json_f:
            data=json.load(json_f)
            img=utils.img_b64_to_arr(data['imageData'])
            #lbl是一个mask，每个obj的像素赋值为该obj对应的value
            lbl, lbl_names_to_value = utils.labelme_shapes_to_label(img.shape, data['shapes'])#包含背景0
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names_to_value)]
            img_with_label=utils.draw_label(lbl,img,captions)

            tmp=[(l-1, name) for l, name in enumerate(lbl_names_to_value) if l!=0]
            lbl_value_to_names={}
            for l,name in tmp:
                lbl_value_to_names[l]=name#不包含背景

            mask=[]
            # instance_id=[]
            for i in range(1,len(lbl_names_to_value)): # 跳过第一个class（默认为背景）
                mask.append((lbl==i).astype(np.uint8)*255) # mask[i]对应第i个对象的mask， 为0、255组成的（0为背景，255为对象）,所有对象的mask叠起来组成三维
                # instance_id.append(i) # mask与clas 一一对应
            # all_mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0]) # 转成[h,w,instance count]
            # instance_id=np.asarray(instance_id,np.uint8) # [instance count,]
    else:
        raise Exception("no such a json file!!")
    
    return img,mask,img_with_label,lbl_value_to_names

def get_singel_obj_mask_bbox(mask):
    '''
    得到某个类别mask的bbox，返回(x1, y1, x2, y2)  
    '''
    no_zero_indexs = np.argwhere(mask)
    if len(no_zero_indexs)>0:
        (y1, x1), (y2, x2) = no_zero_indexs.min(0), no_zero_indexs.max(0) + 1
        return (x1, y1, x2, y2)
    else:
        return None#完全被遮挡则mask为空，bbox设为None
    

def get_region_and_mask(img,mask,bbox):
    '''
    根据bbox返回区域和区域的mask
    '''
    logo_region=img[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
    logo_mask=mask[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
    return logo_region,logo_mask

def copy_to_roi(img,top_left_pt,logo_region,logo_mask):
    '''
    根据参数2的点作为贴图区域的左上角，把图贴到相应位置  
    '''
    logo_region[logo_mask==0]=0
    width,height=logo_region.shape[1],logo_region.shape[0]
    roi=img[top_left_pt[1]:top_left_pt[1]+height,top_left_pt[0]:top_left_pt[0]+width]
    roi[logo_mask!=0]=0
    roi+=logo_region
    return img

def get_random_angle():
    return np.random.randint(1,6)
    
def get_random_location_map():
    return np.random.randint(0,2,(5,7))

def get_n_location_map(n:int):
    if n>35:
        raise Exception("n must lower than 35!!")
    def num2loc(x):
        div=(x-1)//7
        mod=(x-1)%7
        loc=(div,mod)
        return loc
    loc_by_nums=(np.random.choice(35,n,replace=False)+1).tolist()
    loc_map=np.zeros((5,7))
    locs=list(map(num2loc,loc_by_nums))
    for loc in locs:
        loc_map[loc]=1
    return loc_map

def get_location_all(location_map):
    location_row=np.where(location_map!=0)[0][:,np.newaxis]
    location_col=np.where(location_map!=0)[1][:,np.newaxis]
    return np.hstack((location_row,location_col)).tolist(),np.where(location_map!=0)[0].tolist(),np.where(location_map!=0)[1].tolist()
#-------------------------------------
from itertools import groupby
from skimage import measure
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def binary_mask_to_rle(binary_mask):
    '''https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py'''
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons
#----------------------------------
if __name__=="__main__":
    print(get_n_location_map(34))
    # img,all_mask,img_with_label,_=get_all_class_mask("4_6_2.json")
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # # cv2.imshow("test",img)
    # # cv2.waitKey(0)

    # bbox=get_singel_obj_mask_bbox(all_mask[0])
    # x1, y1, x2, y2=bbox
    # logo_region,logo_mask=get_region_and_mask(img,all_mask[0],bbox)
    
    # locations=[(x1,y1-100),(x1,y1-200),(x1,y1-300),(x1,y1-400),(x1+100,y1-200),(x1+200,y1-200),(x1+300,y1-200)]
    # for point in locations:
    #     img=copy_to_roi(img,point,logo_region,logo_mask)
        

    # cv2.imshow("test",img)
    # cv2.imwrite("test_output.jpg",img)
    # cv2.waitKey(0)