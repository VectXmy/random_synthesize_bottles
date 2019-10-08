# random_generate_bottles  
## include      
* bottle.py  bottle类
* synthetic.py 随机生成bottle对象与合成单张图片  
* utils.py   
*  ` scale_xml_bbox.py -d <xml_dir> -s <scale> `  批量按scale缩放pascalvoc格式的xml文件中的xmin,ymin,xmax,ymax四个值   
* generator.py 生成多张图片与coco格式json文件  
* merge_coco_json.py   
## 标签   
    1 -- 'Sprite-Fiber+',   
    2 -- 'PureWater-Nongfu',   
    3 -- 'LemonTea-Vita',   
    4 -- 'MilkTea-Asamu',   
    5 -- 'Cola-bluecan-Pepsi',    
    6 -- 'RedBull-yellowcan-RedBull'   

## 规则  
随机位置，随机饮料，随机角度，5、6限定在中间三列，bbox根据mask遮挡关系自动计算（即只标出未被遮挡的部分）   
## todo  
* 生成json/xml/csv文件的脚本   
* 多线程版本   


## 日志   
* 2019/4/4  修改了utils.copy_to_roi的一个错误；bottle.get_a_bottle增加offset参数以增加瓶子的随机位移（x,y随机在±10内变化）   
* 2019/4/14 添加`generator.py`，同时生成多张图片与coco格式的annotation，存在bug待修复（mask加offset会失败，原因不明，暂且遇到失败则跳过）  
* 2019/4/15  添加merge_coco_json.py,生成主程序见generator.py待完善  
* 2019/4/16  增加step保存   
## 效果  
![test](http://github/VectXmy/random_synthesize_bottles/raw/master/imgs/out1.jpg)   


