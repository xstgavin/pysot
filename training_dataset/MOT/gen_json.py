from os.path import join, isdir
from os import mkdir
import glob
import xml.etree.ElementTree as ET
import json

sub_sets=['train','test']

for sub_set in sub_sets:
    js = {}
    VID_base_path = './MOT2017/'+sub_set
    ann_base_path = join(VID_base_path, 'label.json')
    

    jdata = json.load(open(ann_base_path,'r'))


    n_imgs = len(jdata)
    for f, json_elm in enumerate(jdata.items()):
        print('subset: {} frame id: {:08d} / {:08d}'.format('0', f, n_imgs))
        
        objects = json_elm[1]

        video = json_elm[0].split('.')[0]
        video = video.split('/')[0]+'/'+video.split('/')[1]
        for id, object_iter in enumerate(objects):
            bbox = object_iter['bbox']
            frame = '%06d' %(object_iter['frame'])
            obj = '%02d' % (object_iter['id'])
            visibility = object_iter['visibility']
            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            if visibility <0.3:
                print('vis')
                continue
            js[video][obj][frame]=bbox

    #train = {k:v for (k,v) in js.items() if 'i/' not in k}
    #val = {k:v for (k,v) in js.items() if 'i/' in k}

    json.dump(js, open(sub_set+'.json', 'w'), indent=4, sort_keys=True)

