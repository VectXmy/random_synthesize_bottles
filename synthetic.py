__author__="xu hongtao"
#email: xxxmy@foxmail.com
#wechat: 371802059
#create:03/30/2019

import utils
import numpy as np
import bottle
import cv2
import sys

'''
标签:
    1 -- 'Sprite-Fiber+',
    2 -- 'PureWater-Nongfu',
    3 -- 'LemonTea-Vita',
    4 -- 'MilkTea-Asamu',
    5 -- 'Cola-bluecan-Pepsi',
    6 -- 'RedBull-yellowcan-RedBull'
'''
def _random_bottle(n,jsons_path="./jsons",random_offset=False):
    location_map=utils.get_n_location_map(n)
    # location_map=utils.get_random_location_map()#随机生成bottle的位置矩阵
    # location_map=np.ones((5,7))
    location_all,location_all_row,location_all_col=utils.get_location_all(location_map)#得到所有有bottle的位置坐标,是list，记得后面转换成tuple

    bottles_by_location={}#location:bottle
    bottles_global=[]
    bottles_location_global=[]

    for index,location in enumerate(location_all):
        offset=None
        if random_offset:#每个瓶子添加随机位移
            offset_x=np.random.randint(-10,10)
            offset_y=np.random.randint(-10,10)
            offset=(offset_x,offset_y)

        a_bottle=bottle.get_a_bottle(location,root_json_path=jsons_path,offset=offset)

        sys.stdout.flush()
        print(index,"/",n,end="\r")

        bottles_by_location[tuple(location)]=a_bottle
        bottles_global.append(a_bottle)
        bottles_location_global.append(tuple(location))

    syn_order_map=np.array([[0,0,0,0,0,0,0],#合成从外围到中心
                            [0,1,1,1,1,1,0],
                            [0,1,2,3,2,1,0],
                            [0,1,1,1,1,1,0],
                            [0,0,0,0,0,0,0]])
    subset0=np.argwhere(syn_order_map==0).tolist()
    subset1=np.argwhere(syn_order_map==1).tolist()
    subset2=np.argwhere(syn_order_map==2).tolist()
    subset3=np.argwhere(syn_order_map==3).tolist()
    syn_col_order=(0,1,2,6,5,4,3)#按照列的顺序合成，先两边后中间
    subset0=sorted(subset0,key=lambda x : syn_col_order[x[1]])
    subset1=sorted(subset1,key=lambda x : syn_col_order[x[1]])
    subset2=sorted(subset2,key=lambda x : syn_col_order[x[1]])
    subset3=sorted(subset3,key=lambda x : syn_col_order[x[1]])
    syn_all_order=subset0+subset1+subset2+subset3
    syn_all_order=list(map(tuple,syn_all_order))
    bottles_location_global=sorted(bottles_location_global,key=lambda x:syn_all_order.index(x))

    # syn_col_order=(0,1,2,6,5,4,3)#按照列的顺序合成，先两边后中间
    # #将瓶子分为上下两部分，因为上下的遮挡关系不同会造成修正mask的顺序不同
    # bottles_loaction_global_array=np.array(bottles_location_global)
    # bottles_subset1=bottles_loaction_global_array[np.where(np.array(location_all_row)<2)].tolist()
    # bottles_subset2=bottles_loaction_global_array[np.where(np.array(location_all_row)>=2)].tolist()
    # bottles_subset1=sorted(bottles_subset1,key=lambda x : x[0],reverse=False)
    # bottles_subset2=sorted(bottles_subset2,key=lambda x : x[0],reverse=True)
    # bottles_subset1=sorted(bottles_subset1,key=lambda x : syn_col_order[x[1]])
    # bottles_subset2=sorted(bottles_subset2,key=lambda x : syn_col_order[x[1]])
    # bottles_location_global=bottles_subset1+bottles_subset2
    # # bottles_location_global=sorted(bottles_location_global,key=lambda x : syn_col_order[x[1]])#对瓶子按syn_col_order的顺序排序,em...其实没必要
    # bottles_location_global=list(map(tuple,bottles_location_global))

    #好了，现在我们有瓶子了，开始修正mask并计算bbox，修正mask后可按随意顺序合成
    for xbottle_loc in bottles_location_global:
        bottles_by_location[xbottle_loc].compute_bbox(bottles_by_location)
    return bottles_by_location,bottles_location_global,bottles_global

def _synthesize_img(bg_img,bottles_by_location,bottles_location_global):
    
    bg_img=bg_img.copy()
    for index,xbottle_location in enumerate(bottles_location_global):
        
        xbottle=bottles_by_location[xbottle_location]
        # bbox=utils.get_singel_obj_mask_bbox(xbottle.mask)
        bbox=xbottle.bbox
        if bbox:#有可能完全被遮挡，则bbox为None
            x1, y1, x2, y2=bbox
            bottle_region,bottle_mask=utils.get_region_and_mask(xbottle.img,xbottle.mask,bbox)
            # bottle_region,bottle_mask=xbottle.region,xbottle.region_mask
            bottle_region=cv2.cvtColor(bottle_region,cv2.COLOR_RGB2BGR)
            bg_img=utils.copy_to_roi(bg_img,(x1,y1),bottle_region,bottle_mask)
        
    return bg_img

def get_one_synimg(n,json_path="./jsons",bg_img=None,random_offset=False):
    '''
    Params  
    n 瓶子数量
    bg_img 背景图片  
    Returns  
    bg_img 合成的图片  
    img  画出bbox的结果图  
    '''
    
    if bg_img.any():
        print("----------start synthesizing....----------")
        bottles_by_location,bottles_location_global,bottles_global=_random_bottle(n,jsons_path=json_path,random_offset=random_offset)
        bg_img=_synthesize_img(bg_img,bottles_by_location,bottles_location_global)
        print("----------done！！----------")
    elif bg_img==None:
        bg_img=np.ones((1080,1920,3),dtype=np.uint8)*255
        print("----------start synthesizing....----------")
        bottles_by_location,bottles_location_global,bottles_global=_random_bottle(n,random_offset=random_offset)
        bg_img=_synthesize_img(bg_img,bottles_by_location,bottles_location_global)
        print("----------done！！----------")
    img=bg_img.copy()
    for xbottle in bottles_global:    #框出所有瓶子！
        bbox=xbottle.bbox
        if bbox:
            x1, y1, x2, y2=bbox
            cv2.rectangle(img,(x1,y1),(x2,y2),(250,0,255),1)
    return bg_img,img,bottles_global

if __name__=="__main__":
    #那么现在来合成图片吧！！
    bg_img=cv2.imread("./bg_img.jpg")
    # bg_img=np.zeros((1080,1920,3),dtype=np.uint8)#读取背景图片
    
    bg_img,img_with_bbox,_=get_one_synimg(10,bg_img=bg_img,random_offset=True)
    

    cv2.imshow("test",img_with_bbox)
    # cv2.imwrite("out1.jpg",img_with_bbox)
    cv2.waitKey(0)






