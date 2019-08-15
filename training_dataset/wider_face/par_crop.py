from os.path import join, isdir
from os import mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import time
import sys
import json
# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_xml(xml, sub_set_crop_path, instanc_size=511):
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')

    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)

def crop_json(json_elm, sub_set_crop_path,img_root, instanc_size=511):
    #print('tmp', json_elm[0])
    frame_crop_base_path = join(sub_set_crop_path, json_elm[0].split('.')[0])
    img_path = join(img_root,json_elm[0])
    
    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(json_elm[1]):
        bbox = object_iter['bbox']
        blurness = object_iter['blurness']
        occ = object_iter['occ']
        area = object_iter['area']
        if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)

def main(instanc_size=511, num_threads=24):
    crop_path = './widerFace/crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)
    VID_base_path = './widerFace/'
    sub_sets=['train','val']
    for sub_set in sub_sets:
        ann_base_path = join(VID_base_path, sub_set+'/')
    
        jdata = json.load(open(ann_base_path+'label.json','r'))
        n_imgs = len(jdata)
        print(n_imgs)
        print(len(jdata.items()))
        sub_set_crop_path = crop_path
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_json, json_elm, sub_set_crop_path, './widerFace/'+sub_set,instanc_size) for  json_elm  in jdata.items() ]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix='0', suffix='Done ', barLength=80)

if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
