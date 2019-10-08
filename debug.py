import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f")
args = parser.parse_args()
file_name=args.f

with open(file_name,'r') as fn:
    data=json.load(fn)
    annos=data["annotations"]
    print("images num： ",len(data["images"]))
    seg_num_error=0
    seg_num_good=0
    seg_num_other=0
    bbox_num_error=0
    bbox_num_good=0
    bbox_num_other=0
    for anno in annos:
        seg=anno["segmentation"]
        bbox=anno["bbox"]
        if len(seg)==0:
            seg_num_error+=1
        elif len(seg)>=1:
            seg_num_good+=1
        else:
            seg_num_other+=1

        if len(bbox)==0:
            bbox_num_error+=1
        elif len(bbox)>=1:
            bbox_num_good+=1
        else:
            bbox_num_other+=1
print("segmentation 错误：%s,正确：%s,其他：%s"%(seg_num_error,seg_num_good,seg_num_other))
print("bbox 错误：%s,正确：%s,其他：%s"%(bbox_num_error,bbox_num_good,bbox_num_other))
