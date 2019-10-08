import json


def _merge_img(img_A,img_B):
    num_img_A=len(img_A)
    for obj in img_B:
        obj["id"]+=num_img_A
    img_A.extend(img_B)
    return img_A,num_img_A

def _merge_anno(anno_A,anno_B,num_img_A):
    num_anno_A=len(anno_A)
    for obj in anno_B:
        obj["image_id"]+=num_img_A
        obj["id"]+=num_anno_A
    anno_A.extend(anno_B)
    return anno_A,num_anno_A

def merge_json_data(data_A,data_B):
    anno_A=data_A["annotations"]
    anno_B=data_B["annotations"]
    img_A=data_A["images"]
    img_B=data_B["images"]
    merged_category=data_A["categories"]
    merged_img,num_img_A=_merge_img(img_A,img_B)
    merged_anno,_=_merge_anno(anno_A,anno_B,num_img_A)
    merged_json={}
    merged_json["categories"]=merged_category
    merged_json["images"]=merged_img
    merged_json["annotations"]=merged_anno
    return merged_json
    
def merge_json_file(json_A,json_B):
    fa=open(json_A,'r')
    fb=open(json_B,'r')
    data_A=json.load(fa)
    data_B=json.load(fb)
    merged_data=merge_json_data(data_A,data_B)
    fa.close()
    fb.close()
    return merged_data



if __name__=="__main__":
    
    merged=merge_json_file("./annotations/instances_val2017_id3_full.json","./annotations/instances_val2017_id4_full.json")
    with open("./new.json",'w') as f:
        json.dump(merged,f,indent=4)
    
