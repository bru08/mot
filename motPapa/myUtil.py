"""

"""
import numpy as np
import pandas as pd
import cv2
from os import listdir
import queue
import copy
import itertools


def rect_list_center(rect_list):
    # give a list of rectangles and return a list of the center points
    res = [((x[0] + x[2]) / 2, (x[1] + x[3]) / 2) for x in rect_list]
    return res

def bb_list_center(rect_list):
    # give a list of rectangles and return a list of the center points
    res = [(x[0] + x[2] / 2, x[1] + x[3] / 2) for x in rect_list]
    return res


def get_bb_hw(box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h

def get_bb_pp(box):
    x = box[0]
    y = box[1]
    x2 = box[2] + box[0]
    y2 = box[3] + box[1]
    return x, y, x2, y2


def IoU_calc(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute areas
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class DetectionFileReader:
    # TODO read in mot ch format, also have my detection in mot ch format

    def __init__(self, path, gt = False):
        # import the dataframe with detections
        # acquire max frame number in the file
        self._df = pd.read_csv(path, header=None)
        self._df_last_time = self._df.iloc[-1, 0]
        self._gt = gt

        if self._gt:
            # remove unnecessary columns from mot challenge format file
            # only column 0 and 2,3,4,5 (frameID, box coordinates)
            self._df = self._df.iloc[:, [0, 2, 3, 4, 5]]


    def n_frames(self):
        # return number of frames in the dataframe
        return(self._df_last_time)

    def get_detection(self, frame):
        # return bboxes and centers for detection in a given frame
        assert 0 < frame <= self._df_last_time

        df_slice = self._df.loc[self._df.iloc[:, 0] == (frame), :]
        temp = df_slice.drop(0, axis=1)

        # take just bb(bounding boxes) and ct(the centers
        bbs = temp.values.tolist()
        if self._gt:
            bb_ct = bb_list_center(bbs)
        else:
            bb_ct = rect_list_center(bbs)  # compute centers from list of bboxes

        return bbs, bb_ct  # return bb and centers!



class ImageReader:

    def __init__(self, img_folder_path, loadFull):
        # in_path = "../frames_data/"
        self.images_names = listdir(img_folder_path)
        self.images_names.sort()
        self.images_paths = [img_folder_path + x for x in self.images_names]
        # TODO read next img async
        self.image_buffer = queue.Queue()
        self.buffer_size = 3
        self.proc_index = 0
        self.loadFull = loadFull

        if self.loadFull:
            for elem in self.images_paths:
                self.image_buffer.put(cv2.imread(elem))
            print("frames loaded succesfully!")

    def get_frame(self):
        if not self.loadFull:
            img = cv2.imread(self.images_paths[self.proc_index])
        else:
            img = self.image_buffer.get()

        self.proc_index += 1
        return img


class StuffShower():

    def __init__(self):
        self._exit_keys = [ord("q"), ord("Q"), 27]  # upper and lowcase q and ESC key
        self._circle_radius = 2
        self._id_trajectories = {}
        self._colors = [np.random.randint(0, 255, 3) for i in range(300)]
        self._colors = [[int(x[0]), int(x[1]), int(x[2])] for x in self._colors]
        self._img_memory = []

    def show_detection(self, img, bbs):

        show_clone = copy.copy(img)

        for p in range(len(bbs)):

            x = int(bbs[p][0])
            y = int(bbs[p][1])
            x2 = int(bbs[p][2])
            y2 = int(bbs[p][3])

            cv2.rectangle(show_clone, (x, y), (x2, y2),
                          (255, 0, 0), 2)

        cv2.imshow("tracking", show_clone)

    def show_det_confounding(self, img, bbs, conf_ind):

        show_clone = copy.copy(img)
        conf = list(itertools.chain(*conf_ind))

        for p in range(len(bbs)):
            x = int(bbs[p][0])
            y = int(bbs[p][1])
            x2 = int(bbs[p][2])
            y2 = int(bbs[p][3])

            if p in conf:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(show_clone, (x, y), (x2, y2),
                          color, 2)

        cv2.imshow("tracking", show_clone)



    def show_tracking(self, img, bbs, cts, ids):

        show_clone = copy.copy(img)

        for p in range(len(bbs)):

            self.update_trajectories(ids[p], cts[p])
            x = int(bbs[p][0])
            y = int(bbs[p][1])
            x2 = int(bbs[p][2])
            y2 = int(bbs[p][3])

            x_c = int(cts[p][0])
            y_c = int(cts[p][1])

            cv2.rectangle(show_clone, (x, y), (x2, y2),
                          self._colors[ids[p]], 2)

            # for elem in self._id_trajectories[ids[p]]:
            #     cv2.circle(show_clone, (int(elem[0]), int(elem[1])), self._circle_radius,
            #                (255, 0, 0), -1)
            if len(self._id_trajectories[ids[p]]) > 1:
                cv2.polylines(show_clone, [np.int32(self._id_trajectories[ids[p]])],
                              False, self._colors[ids[p]], 2)

            msg = str(ids[p])
            fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            fontScale = .5
            thickness = 2
            textSize = cv2.getTextSize(msg, fontFace, fontScale, thickness)

            cv2.rectangle(show_clone, (x_c, y_c - textSize[1]-3), (x_c + textSize[0][0], y_c - textSize[1] - textSize[0][1] - 3),
                          (255,255,255), -1)

            cv2.putText(show_clone, msg,
                        (x_c, y_c - 10),
                        fontFace, fontScale,
                        (0, 0, 0), thickness)  # Write the prediction class
        self._img_memory.append(copy.copy(show_clone))
        cv2.imshow("tracking", show_clone)

    def update_trajectories(self, id, ct):

        if id in self._id_trajectories.keys():
            self._id_trajectories[id].append(ct)
        else:
            self._id_trajectories[id] = [ct]


    def show_wait(self, t):
        if cv2.waitKey(t) in self._exit_keys:
            cv2.destroyAllWindows()
            return False
        else:
            return True

    def save_video(self, path):
        h, w, lay = self._img_memory[0].shape

        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

        for i in range(len(self._img_memory)):
            out.write(self._img_memory[i])

        out.release()

