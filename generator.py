__author__="xu hongtao"
#email: xxxmy@foxmail.com
#wechat: 371802059
#create: 04/14/2019

import synthetic
import utils
import json
import cv2
import numpy as np
# from merge_coco_json import merge_json
import sys
import os


coco_json_categories = [
    {
        'id': 1,
        'name':'Sprite-Fiber+',
        'supercategory': 'drink',
    },
    {
        'id': 2,
        'name': 'PureWater-Nongfu',
        'supercategory': 'drink',
    },
    {
        'id': 3,
        'name': 'LemonTea-Vita',
        'supercategory': 'drink',
    },
    {
        'id': 4,
        'name': 'MilkTea-Asamu',
        'supercategory': 'drink',
    },
    {
        'id': 5,
        'name': 'Cola-bluecan-Pepsi',
        'supercategory': 'drink',
    },
    {
        'id': 6,
        'name': 'RedBull-yellowcan-RedBull',
        'supercategory': 'drink',
    },
]

# def out_coco_json(train_or_val,id):
#     coco_json = {
#             # "info": INFO,
#             # "licenses": LICENSES,
#             "categories": coco_json_categories,
#             "images": coco_json_images,
#             "annotations": coco_json_annotation
#         }
#     if train_or_val=="val":
#         with open("./data_%s/annotations/"%(str(id))+'instances_val2017_id%s_full.json'%(str(id)),'w') as f :
#             json.dump(coco_json,f,indent=4)
#     elif train_or_val=="train":
#         with open("./data_%s/annotations/"%(str(id))+'instances_train2017_id%s_full.json'%(str(id)),'w') as f :
#             json.dump(coco_json,f,indent=4)

def create_dir(id):
    if not os.path.exists("./data_%s"%(str(id))):
        os.mkdir("./data_%s"%(str(id)))
        os.mkdir("./data_%s/images"%(str(id)))
        os.mkdir("./data_%s/annotations"%(str(id)))
        os.mkdir("./data_%s/images/train2017"%(str(id)))
        os.mkdir("./data_%s/images/val2017"%(str(id)))
        print("--------------./data_%s has been created"%(str(id)))
    else:
        print("--------------./data_%s exist"%(str(id)))

def main(num,id,step,per_num,train_or_val="train",bg_img=None,random_offset=True,jsons_path="./jsons"):
    '''
    #第一个参数为生成数量  
    #第二个参数，为生成者id用以区分不同生成的批次，比如两次生成后来合并成一个 
    #第三个参数step表示多少张图片保存一次json，防止中间故障导致前功尽弃 
    #第四个参数表示每张图瓶子的数量
    #train_or_val “train”为生成训练集，“val”为生成验证集  
    #bg_img 为背景图片  
    #jsons_path jsons目录  
    #random_offset 为True则加入随机位移  
    # '''  
    create_dir(id)
    coco_json_images=[]
    coco_json_annotation=[]
    for index in range(num):
        try:
            img,_,bottles_global=synthetic.get_one_synimg(per_num,json_path=jsons_path, bg_img=bg_img,random_offset=random_offset)
            print(train_or_val," 进度： ",index+1,"/",num,"  id : ",str(id))
        except Exception as e:
            print("------------这个失败了.....  ",e)
            continue
        images_el={}
        images_el["height"]=img.shape[0]
        images_el["width"]=img.shape[1]
        images_el["id"]=len(coco_json_images)+1
        images_el["file_name"]=str(len(bottles_global))+"_"+str(index)+"_"+str(id)+".jpg"#命名为<瓶子数量>_<序号>_<生成者id>.jpg
        coco_json_images.append(images_el)
        if train_or_val=="val":
            cv2.imwrite("./data_%s/images/val2017/"%(str(id))+images_el["file_name"],img)
        elif train_or_val=="train":
            cv2.imwrite("./data_%s/images/train2017/"%(str(id))+images_el["file_name"],img)
        else:
            raise TypeError("train_or_val must in [\"train\",\"val\"]")

        for xbottle_id,xbottle in enumerate(bottles_global): 
            if xbottle.bbox!=None :
                xbottle_annotation={}
                xbottle_annotation["segmentation"]=utils.binary_mask_to_polygon(xbottle.mask,tolerance=2)
                if len(xbottle_annotation["segmentation"])>=1:
                    xbottle_annotation["iscrowd"]=0
                    xbottle_annotation["image_id"]=images_el["id"]
                    x1,y1,x2,y2=xbottle.bbox
                    w,h=x2-x1,y2-y1
                    xbottle_annotation["bbox"]=[int(x1),int(y1),int(w),int(h)]
                    xbottle_annotation["category_id"]=xbottle.bottle_class
                    xbottle_annotation["id"]=len(coco_json_annotation)+1
                    # binary_mask_encoded = encode_mask.encode(np.asfortranarray(xbottle.mask.astype(np.uint8)))
                    xbottle_annotation["area"]=len(np.argwhere(xbottle.mask!=0))
                    coco_json_annotation.append(xbottle_annotation)
                else:
                    print("---seg 转换错误，跳过第%s个瓶子---"%(str(xbottle_id)))

        if index%step==0 and index!=0 and index!=num:
            tmp_json = {
                "categories": coco_json_categories,
                "images": coco_json_images,
                "annotations": coco_json_annotation
            }
            if train_or_val=="val":
                with open("./data_%s/annotations/"%(str(id))+'instances_val2017_id%s_step%s.json'%(str(id),str(index)),'w') as f :
                    json.dump(tmp_json,f,indent=4)
            elif train_or_val=="train":
                with open("./data_%s/annotations/"%(str(id))+'instances_train2017_id%s_step%s.json'%(str(id),str(index)),'w') as f :
                    json.dump(tmp_json,f,indent=4)
    coco_json = {
            # "info": INFO,
            # "licenses": LICENSES,
            "categories": coco_json_categories,
            "images": coco_json_images,
            "annotations": coco_json_annotation
        }
    if train_or_val=="val":
        with open("./data_%s/annotations/"%(str(id))+'instances_val2017_id%s_full.json'%(str(id)),'w') as f :
            json.dump(coco_json,f,indent=4)
    elif train_or_val=="train":
        with open("./data_%s/annotations/"%(str(id))+'instances_train2017_id%s_full.json'%(str(id)),'w') as f :
            json.dump(coco_json,f,indent=4)


if __name__=="__main__":
    import argparse

    parser=argparse.ArgumentParser(description="generate images")
    parser.add_argument("--num")
    parser.add_argument("--id",type=str)
    parser.add_argument("--save_step")
    parser.add_argument("--per_img_num")
    parser.add_argument("--train_or_val",type=str)
    parser.add_argument("--jsons",default="./jsons")
    parser.add_argument("--offset",action='store_true')

    args=parser.parse_args()
    num=args.num
    creator_id=args.id
    step=args.save_step
    per_num=args.per_img_num
    train_or_val=args.train_or_val
    jsons_dir=args.jsons
    offset=args.offset

    main(num,creator_id,step,per_num,train_or_val=train_or_val,bg_img=bg_img,jsons_path=jsons_dir,random_offset=offset)
    bg_img=cv2.imread("./bg_img.jpg")
    # main(2000,"419_1",2000,15,train_or_val="train",bg_img=bg_img,jsons_path="./jsons",random_offset=True)
    # main(20,"423_35_1",2000,35,train_or_val="val",bg_img=bg_img,jsons_path="./jsons",random_offset=True)
