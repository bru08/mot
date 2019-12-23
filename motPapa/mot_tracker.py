"""

"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from .myUtil import get_bb_hw




class TrackerByDetection:

    def __init__(self, image_array, init_bb, init_ct):

        self._img_array = image_array  # pointer to the image in memory
        self._id_dict = {}
        self._max_id = 0
        self._bb_buffer = [init_bb]
        self._ct_buffer = [init_ct]
        self._dormient_length = 10  # n frame for which we store inactive id before discarding them
        self.ids = []

        temp = []
        for elem in list(zip(init_bb, init_ct)):
            self._max_id += 1
            self._id_dict[self._max_id] = -1  # n frames of inactivity
            temp.append(self._max_id)

        self.ids.append(temp)

    def update(self, bb, ct):

        self._bb_buffer.append(bb)
        self._ct_buffer.append(ct)

    def IoU_calc(self, boxA, boxB):

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

    def lap_constrained(self):

        points_a = self._ct_buffer[-2]
        points_b = self._ct_buffer[-1]

        bb_a = self._bb_buffer[-2]
        bb_b = self._bb_buffer[-1]

        dist_mat = distance_matrix(points_a, points_b)
        a_ind, b_ind = linear_sum_assignment(dist_mat)

        poss_assignment = list(zip(a_ind, b_ind))
        valid_assignments = []

        for elem in poss_assignment:
            c1 = points_a[elem[0]]
            c2 = points_b[elem[1]]

            # take height and width of the two rectangles

            b1_hw = get_bb_hw(bb_a[elem[0]])[2:]
            b2_hw = get_bb_hw(bb_b[elem[1]])[2:]

            mvmt_threshold = int(np.max([*b1_hw, *b2_hw]))
            valid_displacement_flag = (euclidean(c1, c2) < mvmt_threshold)

            if valid_displacement_flag:
                valid_assignments.append(elem)

        return valid_assignments  # old index to new index in detections

    def check_confounders(self):
        # check bb in current frame that may occlude themselves in future frame
        # if iou > 0
        current_bbs = self._bb_buffer[-1]
        current_ct = self._ct_buffer[-1]

        confounders = []
        for i in range(len(current_bbs)):
            bb_hw0 = get_bb_hw(current_bbs[i])[2:]
            temp = [i]
            for j in range(i + 1, len(current_bbs)):
                bb_hw1 = get_bb_hw(current_bbs[j])[2:]

                proximity_th = 2 * np.max([np.min([*bb_hw0]), np.min([*bb_hw1])])

                dist = euclidean(current_ct[i], current_ct[j])

                too_close = (dist < proximity_th)
                if too_close:
                    # print("th: ", proximity_th, "dist", dist)
                    temp.append(j)
            if len(temp) > 1:
                confounders.append(temp)  # append found cluster of close obj

        return confounders

    def update_id(self):

        for k in self._id_dict.keys():
            self._id_dict[k] += 1

        assign = self.lap_constrained()
        old_ids = self.ids[-1]
        old_ref = [x[0] for x in assign]
        new_ref = [x[1] for x in assign]

        players = [x for x in range(len(self._bb_buffer[-1]))]

        temp = []
        for i in players:

            if i in new_ref:
                pos = new_ref.index(i)
                new_ind = old_ids[old_ref[pos]]
                temp.append(new_ind)
                self._id_dict[new_ind] = -1
            else:

                possible = [k for k, v in self._id_dict.items() if v > 0]

                self._max_id += 1
                self._id_dict[self._max_id] = -1
                temp.append(self._max_id)

        self.ids.append(temp)

    def get_present_ids(self):
        return self.ids[-1]
