import sys
import os

if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
else:
    gpu_id = "gpu0"
    cnmem = "0.3"
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem
from batch_generators import load_names, load_images
from models import net
import numpy as np
import cv2
import time
import warnings
from math import sqrt
from matplotlib import cm
warnings.filterwarnings("ignore")


def checkBadFrame(sequence, frame):

    if sequence == 'data_04-24-38':
        if frame in range(32, 64) or frame in range(268, 273):
            return True

    if sequence == 'data_03-54-47':
        if frame in range(1, 8) or frame in range(14, 23):
            return True

    if sequence == 'data_04-21-34':
        if frame in range(70, 84):
            return True

    if sequence == 'data_04-20-27':
        if frame in range(165, 168):
            return True

    if sequence == 'data_04-15-01':
        if frame in range(108, 110):
            return True

    if sequence == 'data_03-31-37':
        if frame in [149, 151, 39, 40, 50, 30, 31, 36, 37, 38]:
            return True

    if sequence == 'data_04-22-13':
        if frame in range(76, 86) or frame in range(135, 140):
            return True

    if sequence == 'data_03-31-52':
        if frame in range(89, 103):
            return True

    if sequence == 'data_04-17-41':
        if frame in range(76, 85):
            return True

    if sequence == 'data_04-21-43':
        if frame in range(100, 105) or frame in range(57, 81) or frame in range(21, 34):
            return True

    if sequence == 'data_04-36-55':
        if frame in range(130, 137):
            return True

    if sequence == 'data_04-34-45':
        if frame in range(191, 194):
            return True

    if sequence == 'data_04-25-52':
        if frame in range(130, 137):
            return True

    if sequence == 'data_04-21-26':
        if frame in range(146, 150):
            return True

    return False


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if (xB - xA ) < 0 or (yB - yA )<0:
        return 0
    interArea = (xB - xA ) * (yB - yA )
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def get_range(img, x, y, kernelsize=15):
    F = 365.337
    R = 245.0
    minus = (kernelsize - 1) / 2
    plus = (kernelsize / 2) + 1
    box_Range = img[x - minus:x + plus, y - minus:y + plus]
    box_mean = (np.mean(box_Range))
    try:
        range = int(round(F * R / float(box_mean)))
    except:
        range = 30

    return range

if __name__ == '__main__':

    # image parameters
    rows = 424
    cols = 512
    ch = 1

    # training parameters
    b_crop = False
    b_rescale = True
    b_scale = True
    b_normcv2 = False

    batch_size = 2

    # image visualization
    b_visualize = True

    # graph visualization
    b_plot = True
    save = True

    # weights
    pesi = '../weights_13_cpm_no_vgg_out_8BIT/weights.041-0.00920.hdf5'

    # model
    model = net(input_shape=(1, rows, cols), weights_path=pesi)

    # loading test names
    test_data_names = load_names(val_seq=2, augm=0, dataset=2)
    show = True
    thresold = 0.3
    TP = TN = FP = FN = FP_DIST = 0
    FPS = []
    TOTiou = []
    for image in range(len(test_data_names)):

        sys.stdout.write("\r%.2f%%" % (((image + 1) / float(len(test_data_names))) * 100))
        sys.stdout.flush()
        seq = test_data_names[image]['image'].split('\\')[-3]
        frame = 0
        if checkBadFrame(seq,frame):
            print 'skipped', seq, frame
            continue
        if test_data_names[image]['face'] == 0:
           print 'SKIP NO GT', seq, frame
           continue
        t = time.time()

        test_data_X, _ = load_images(test_data_names[image:image+1], crop=b_crop, rescale=b_rescale, scale=b_scale, b_debug=False,
                                     normcv2=b_normcv2, rows=rows, fulldepth=False, cols=cols, equalize=True,
                                     removeBackground=True,division=4)

        pred = model.predict(x=test_data_X, batch_size=batch_size, verbose=0)
        for index,i in enumerate(pred):
            gt_head = bool(test_data_names[image+index]['face'])
            if gt_head:
                gt_coord = test_data_names[image + index]['facecord']
            head = False
            img = cv2.imread(test_data_names[image+index]['image'], cv2.IMREAD_ANYDEPTH)
            img = cv2.resize(img,(cols,rows))
            gt = img.copy()
            gt = (gt - 500) / 8.0
            gt = gt.astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGRA)
            merge = gt.copy()
            i = cv2.resize(i, (cols,rows))
            hit = np.where(i[:,:,0]>thresold)
            hitbox = []
            for hx,hy in zip(hit[0],hit[1]):
                r = get_range(img,hx,hy)
                hitbox.append((hx-r/2,hy-r/2,hx+r/2,hy+r/2))
            hitbox = np.asarray(hitbox)
            bestbox = non_max_suppression_fast(hitbox,0.2)
            t = time.time() - t
            for b in bestbox:
                cv2.rectangle(gt, (b[1], b[0]), (b[3], b[2]), (0, 255, 0), 1)
                head=True
            head_count = len(bestbox)
            gt_bb = []
            if gt_head:
                dist = 0
                for el in gt_coord:
                    r = get_range(img,el[1],el[0])
                    if show:
                        cv2.rectangle(gt, (el[0] - r / 2, el[1] - r / 2), (el[0] + r / 2, el[1] + r / 2), (255, 0, 0),1)
                    gt_bb.append((el[1]-r/2,el[0]-r/2,el[1]+r/2,el[0]+r/2))
                if head:
                    iou = []
                    for b in gt_bb:
                        bestindex = -1
                        bestvalue = 9999
                        for bbindex in range(len(bestbox)):
                            value = sqrt((b[0]-bestbox[bbindex][0])**2+(b[1]-bestbox[bbindex][1])**2)
                            if value<bestvalue:
                                bestindex = bbindex
                                bestvalue = value
                        iou.append(bb_intersection_over_union(bestbox[bestindex],b))
                        TOTiou.append(bb_intersection_over_union(bestbox[bestindex],b))
                    for indexel,el in enumerate(iou):
                        print ' iou {} <- head {} '.format(el,indexel)
            if show:
                if image % 1 == 0:
                    cv2.imwrite('result\{}_bb.jpg'.format(image), gt)
                cv2.imshow('GT', gt)
                cv2.waitKey(1)
                i = (i[:, :, 0] * 255).astype(np.uint8)
                i = cv2.applyColorMap(i, cv2.COLORMAP_JET)
                i = cv2.cvtColor(i, cv2.COLOR_BGR2BGRA)

            for headindex in range(head_count):
                if head:
                    if gt_head:
                        if headindex >= len(iou):
                            FP +=1
                            continue
                        if iou[headindex] > 0.3:
                            TP += 1
                        else:
                            FP_DIST +=1
                    else:
                        FP += 1
                else:
                    if gt_head:
                        FN += 1
                    else:
                        TN += 1
            if show:
                alpha = 0.5
                cv2.addWeighted(i, alpha, merge, 1 - alpha, 0, merge)
                if image%1 == 0:
                    cv2.imwrite('result\{}_map.jpg'.format(image), merge)
                cv2.imshow('MERGE', merge)
                cv2.waitKey(1)

            FPS.append(1/t)
            if image % 100 == 0:
                print '\ntp: {} fp: {} fp_iou: {} tn: {} fn: {} ||| FPS: {}'.format(TP, FP, FP_DIST, TN, FN, np.mean(FPS))
    print '\ntp: {} fp: {} fp_iou: {} tn: {} fn: {} ||| FPS: {}'.format(TP,FP,FP_DIST,TN,FN,np.mean(FPS))
    print 'tp-rate: {}'.format(float(TP)/(FP_DIST+TP+FP+FN+TN)*100)
    print 'tn-rate: {}'.format(float(TN)/(FP_DIST+TP+FP+FN+TN)*100)
    print 'average iou: {}'.format(np.mean(TOTiou))
    with open(pesi.split('/')[0]+'TEST_result.txt','w') as out:
        out.writelines('tp: {} fp: {} fp_iou: {} tn: {} fn: {} ||| FPS: {}'.format(TP,FP,FP_DIST,TN,FN,np.mean(FPS)))
        out.writelines('\ntp-rate: {}'.format(float(TP) / (FP_DIST + TP + FP + FN + TN) * 100))
        out.writelines('\ntn-rate: {}'.format(float(TN) / (FP_DIST + TP + FP + FN + TN) * 100))
        out.writelines('\naverage iou: {}'.format(np.mean(TOTiou)))