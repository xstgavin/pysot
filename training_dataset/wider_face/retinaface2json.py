import json
import numpy



sub_sets=['train','val']
for sub_set in sub_sets:
    count=0
    fileName='./widerFace/'+sub_set+'/label.txt'
    lines = open(fileName,'r').readlines()

    fp_bbox_map={}
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            if count!=0 and len(fp_bbox_map[name])==0:
                fp_bbox_map.pop(name,None)
            name = 'images/'+line[1:].strip()
            fp_bbox_map[name] = []
            count = count + 1
            continue
        assert name is not None
        assert name in fp_bbox_map
        values = [float(x) for x in line.strip().split()]
        #if values[19]<0.3:
        #    continue
        bbox = [values[0], values[1], values[0]+values[2], values[1]+values[3]]
        elm={}
        elm['bbox']=bbox
        elm['area']=values[2]*values[3]
        if elm['area']<400:
            continue
        if sub_set=='val':
            fp_bbox_map[name].append(elm)
            continue
   
        blur= values[19]
        elm['blurness']=blur
        elm['occ'] = (values[6]+values[9]+values[12]+values[15]+values[18])/5.0
        if elm['blurness']<0.3:
            continue
        fp_bbox_map[name].append(elm)

    json.dump(fp_bbox_map, open('./widerFace/'+sub_set+'/label.json', 'w'), indent=4, sort_keys=True)
