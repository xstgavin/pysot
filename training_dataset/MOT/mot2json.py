import json
import numpy
import glob
sub_sets=['train','test']
for sub_set in sub_sets:
    count=0
    folderName='./MOT2017/'+sub_set
    folders = glob.glob(folderName+'/*')
    fp_bbox_map={}
   
    for folder in folders:
        if folder.split('-')[-1]!='FRCNN':
            continue
        lines = open(folder+'/gt/gt.txt','r').readlines()
        imgfolder=folder+'/img1/'
        for line in lines:
            imgName = folder.split('/')[-1]+'/img1/'+'%06d'%int(line.split(',')[0])+'.jpg'
            if imgName not in fp_bbox_map:
               fp_bbox_map[imgName]=[]
            
            values = [float(x) for x in line.strip().split(',')]
            bbox = [values[2], values[3], values[2]+values[4], values[3]+values[5]]
            elm={}
            elm['bbox']=bbox
            elm['frame']= int(values[0])
            elm['id']= int(values[1])
            elm['conf_score']=values[6]
            elm['class']=int(values[7])-1
            elm['visibility'] = values[8]
            fp_bbox_map[imgName].append(elm)
    json.dump(fp_bbox_map, open('./MOT2017/'+sub_set+'/label.json', 'w'), indent=4, sort_keys=True)
