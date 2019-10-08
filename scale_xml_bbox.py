#  @xxxmy@foxmail.com


import xml.etree.ElementTree as ET
import os
import argparse

def scale_bbox(input_xml,scale,output_xml=None): 
    tree = ET.parse(input_xml)
    root = tree.getroot() 

    for element in root.findall('object/bndbox'):
        xmin = element.find('xmin')  # 访问Element文本
        ymin = element.find('ymin')
        xmax = element.find('xmax')
        ymax = element.find('ymax')
        xmin_text=xmin.text
        ymin_text=ymin.text
        xmax_text=xmax.text
        ymax_text=ymax.text
        bbox_para=[]
        bbox_obj=[]
        bbox_para.extend([xmin_text,ymin_text,xmax_text,ymax_text])
        bbox_obj.extend([xmin,ymin,xmax,ymax])
        for obj,para in zip(bbox_obj,bbox_para):
            newtext=str(int(float(para)*scale))
            obj.text=newtext
    if output_xml==None:
        output_xml=input_xml
    tree.write(output_xml)

def scale_xml_bbox(xml_dir,scale):
    xml_flies=os.listdir(xml_dir)
    xml_flies=[x for x in xml_flies if x.endswith(".xml")]
    num=len(xml_flies)
    done=0
    for file_ in xml_flies:
        if file_.endswith(".xml"):
            path=os.path.join(xml_dir,file_)
            scale_bbox(path,scale)
            done+=1
            print(done,"/",num)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='scale xml bbox')
    parser.add_argument("-s",'--scale', help='scale')
    parser.add_argument("-d",'--xml_dir', help='xml dir')
    args = parser.parse_args()
    scale_xml_bbox(args.xml_dir,float(args.scale))

