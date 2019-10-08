__author__="xu hongtao"
#email: xxxmy@foxmail.com
#wechat: 371802059
#create:03/30/2019

import numpy as np
import utils
import cv2


def get_a_bottle(location,root_json_path="./jsons",offset=None):
    '''
    根据location获得一个随机bottle对象  
    parameters：  
    location   位置矩阵中的坐标，（row,col）  
    '''
    col=location[1]#所在列

    angle=np.random.randint(1,6)
    bottle_class=np.random.randint(1,7)
    
    if (bottle_class==5 or bottle_class==6) and (col!=2 and col!=3 and col!=4):#如果是罐装在靠外的列，则换种饮料，也就是说罐装只能在中间
        bottle_class=np.random.randint(1,5)
    # if (bottle_class!=5 and bottle_class!=6) and (col==2 or col==3 or col==4):#如果是罐装在靠外的列，则换种饮料，也就是说罐装只能在中间
    #     bottle_class=np.random.randint(5,7)

    location_by_num=location[0]*7+location[1]+1

    # if not json_file:
    json_file=root_json_path+"/"+ str(bottle_class)+"_"+str(location_by_num)+"_"+str(angle)+".json"

    img,all_mask,_,_=utils.get_all_class_mask(json_file)
    mask=all_mask[0]

    if offset!=None:
        bbox=utils.get_singel_obj_mask_bbox(mask)
        bottle_region,bottle_mask=utils.get_region_and_mask(img,mask,bbox)
        mask_region,mask_mask=utils.get_region_and_mask(mask,mask,bbox)

        #bbox加上offset
        offset_x,offset_y=offset[0],offset[1]
        bbox=(min(bbox[0]+offset_x,img.shape[1]),min(bbox[1]+offset_y,img.shape[0]),min(bbox[2]+offset_x,img.shape[1]),min(bbox[3]+offset_y,img.shape[0]))

        new_mask=np.zeros_like(mask)#mask加上offset
        mask=utils.copy_to_roi(new_mask,(bbox[0],bbox[1]),mask_region,mask_mask)
        # cv2.imwrite("mask.jpg",mask)

        
        img=utils.copy_to_roi(img,(bbox[0],bbox[1]),bottle_region,bottle_mask)#img加上offset
        # cv2.imwrite("img.jpg",img)

    return bottle(img,tuple(location),angle,mask,bottle_class)




class bottle(object):
    def __init__(self,img,location,angle,mask,bottle_class):
        self.img=img
        self.location=location
        self.angle=angle
        self.mask=mask#整张图的mask
        self.bbox=None
        self.bottle_class=bottle_class
        # self.region=bottle_region#原图截取的瓶子区域
        # self.region_mask=bottle_mask#原图截取的瓶子区域的mask
        

    def modify_mask(self,bottles_by_location):
        all_locs=bottles_by_location.keys()
        row=self.location[0]
        col=self.location[1]
        #与8领域的bottle去除被遮挡部分，修正mask
        domain_loc=[]
        left_b_loc,right_b_loc,top_b_loc,bottom_b_loc=(row,col-1),(row,col+1),(row-1,col),(row+1,col)    
        left_top_b_loc,left_bottom_b_loc,right_top_b_loc,right_bottom_b_loc=(row-1,col-1),(row+1,col-1),(row-1,col+1),(row+1,col+1)
        special_b_loc=[(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,2),(3,3),(3,4)]
        domain_loc.append(left_b_loc)
        domain_loc.append(right_b_loc)
        domain_loc.append(top_b_loc)
        domain_loc.append(bottom_b_loc)
        domain_loc.append(left_top_b_loc)
        domain_loc.append(left_bottom_b_loc)
        domain_loc.append(right_top_b_loc)
        domain_loc.append(right_bottom_b_loc)
        special_b_loc=list(set(special_b_loc)-set(domain_loc))
        for candidate_loc in domain_loc:
            if candidate_loc in all_locs :
                candidate_bottle=bottles_by_location[candidate_loc]
                inter_mask=cv2.bitwise_and(self.mask,candidate_bottle.mask)
                self.mask=self.mask-inter_mask
        for candidate_loc in special_b_loc:#中间3*3位置的瓶子也可能对其他位置有遮挡
            if candidate_loc in all_locs and self.location not in special_b_loc:
                candidate_bottle=bottles_by_location[candidate_loc]
                inter_mask=cv2.bitwise_and(self.mask,candidate_bottle.mask)
                self.mask=self.mask-inter_mask
        

    
    def compute_bbox(self,bottles_by_location):
        
        self.modify_mask(bottles_by_location)
        self.bbox=utils.get_singel_obj_mask_bbox(self.mask)