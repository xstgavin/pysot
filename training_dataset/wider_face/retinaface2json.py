import json
import numpy



sub_sets=['train','val']
for sub_set in sub_sets:
    fileName='./'+sub_set+'/label.txt'
    lines = open(fileName,'r').readlines()

    fp_bbox_map={}
    for line in lines[:1000]:
        line = line.strip()
        if line.startswith('#'):
            name = 'images/'+line[1:].strip()
            fp_bbox_map[name] = []
            continue
        assert name is not None
        assert name in fp_bbox_map
        values = [float(x) for x in line.strip().split()]
        #if values[19]<0.3:
        #    continue
        bbox = [values[0], values[1], values[0]+values[2], values[1]+values[3]]
        blur= values[19]
        elm={}
        elm['bbox']=bbox
        elm['blurness']=blur
        elm['occ'] = (values[6]+values[9]+values[12]+values[15]+values[18])/5.0
        elm['area']=values[2]*values[3]
        if elm['area']<400 or elm['blurness']<0.3:
            continue
        fp_bbox_map[name].append(elm)

    json.dump(fp_bbox_map, open('./'+sub_set+'/label.json', 'w'), indent=4, sort_keys=True)