import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f")
args = parser.parse_args()
file_name=args.f
# file_name="./data_419_1/annotations/instances_train2017_id419_1_full.json"

with open(file_name,'r') as f:
    data =json.load(f)
    annos=data["annotations"]
    images=data["images"]
    categories=data["categories"]

    new_annos=[]

    for anno in annos:
        seg=anno["segmentation"]
        if len(seg)>=1:
            new_annos.append(anno)
    data["annotations"]=new_annos

    with open("new_modified.json",'w') as fn:
        json.dump(data,fn,indent=4)
    print("done")