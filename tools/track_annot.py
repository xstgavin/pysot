from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
from torch import nn
import numpy as np
from glob import glob
import json

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    #logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # create model
    model = ModelBuilder()
    
    #model = nn.DataParallel(model)
    # load model
    print(args.snapshot,"model path")
    #print(torch.load(args.snapshot))
    pretrained_path=args.snapshot
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cpu())
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict)
    print("torch load done")

    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    print("model.load_state_dict done")
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    tracker2 = build_tracker(model)
    first_frame = True
    video_track_result =  args.video_name+'.pysot'
    fid = open(video_track_result,'w')
    trackInfo={}
    trackInfo['vidName']=args.video_name
    trackInfo['ids']=[] 
    while(True):
        gotNew=int(input('new_trackor? '))
        if gotNew <0:
            break
        idInfo={}
        idInfo['id']=gotNew
        idInfo['trackInfo'] = []
        first_frame = True
        if args.video_name:
            video_name = args.video_name.split('/')[-1].split('.')[0]
        else:
            video_name = 'webcam'
        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        frameIdx = 0
        for frame in get_frames(args.video_name):
            frameIdx = frameIdx + 1
            if first_frame:
              skip = int(input('is_skip?: '))
            if first_frame and skip==0 :
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    print(init_rect)
                    dat = [frameIdx,init_rect[0],init_rect[1],init_rect[2],init_rect[3]]
                    print(dat)
                    idInfo['trackInfo'].append(dat)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            elif first_frame  and  skip==1:
                cv2.imshow(video_name,frame)
                cv2.waitKey(40)
                continue
            else:
                #outputs = tracker.track(frame)
                outputs = tracker.track(frame)#,'track1_%d.jpg'%frameIdx)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (0, 255, 0), 3)
                dat = [frameIdx,bbox[0],bbox[1],bbox[2],bbox[3]]
                idInfo['trackInfo'].append(dat)
                cv2.imshow(video_name, frame)
                cv2.waitKey(40)
        trackInfo['ids'].append(idInfo)
    json.dump(trackInfo,fid,indent=4)
    fid.close() 
if __name__ == '__main__':
    main()
