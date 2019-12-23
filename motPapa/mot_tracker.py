"""

"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean, cosine
from .myUtil import get_bb_hw
import cv2




class TrackerByDetection:

    def __init__(self, image_array, init_bb, init_ct):

        self._img_array = image_array  # pointer to the image in memory
        self._id_dict = {}
        self._max_id = 0
        self._bb_buffer = [init_bb]
        self._ct_buffer = [init_ct]
        self._dormient_length = 10  # n frame for which we store inactive id before discarding them
        self.ids = []
        self._id_data = {}  # dict to store data from ids location, histograms for colors etc
        self._max_recovery_dist = 160
        self._max_recovery_delay = 50
        self._id_hoc = {}
        self._mov_diff_th = 70
        self._max_recovery_cos = 0.1

        temp = []
        for elem in list(zip(init_bb, init_ct)):
            self._max_id += 1
            self._id_dict[self._max_id] = {"status":-1,
                                           "ct":elem[1]}# n frames of inactivity
            temp.append(self._max_id)

        self.ids.append(temp)

    def update(self, frame, bb, ct):

        self._bb_buffer.append(bb)
        self._ct_buffer.append(ct)
        self._img_array.append(frame)


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
            self._id_dict[k]["status"] += 1

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
                self._id_dict[new_ind]["status"] = -1
                self._id_dict[new_ind]["ct"] = self._ct_buffer[-1][i]
            else:

                ret, min_id = self.proximity_recovery(i)
                if ret:
                    temp.append(min_id)

                if not ret:
                    self._max_id += 1
                    self._id_dict[self._max_id] = {"status": -1,
                                                    "ct": self._ct_buffer[-1][i]}
                    temp.append(self._max_id)

        self.ids.append(temp)

    def proximity_recovery(self, i):

        possible = [k for k, v in self._id_dict.items() if v["status"] > 0]
        ct = self._ct_buffer[-1][i]  # position of new detection

        min_cost = self._max_recovery_dist
        min_id = None
        for k in possible:
            dist = euclidean(ct, self._id_dict[k]["ct"])
            if dist < min_cost:
                min_id = k
                min_cost = dist

        if min_id is not None and self._id_dict[min_id]["status"] < self._max_recovery_delay:
            print("recovered id: ", min_id, "last dist", min_cost)
            self._id_dict[min_id]["status"] = -1
            self._id_dict[min_id][ct] = ct
            return True, min_id
        else:
            return False, -1


    def get_hoc(self):
        # take hoc of id persistent between this and previous frame
        # id qui e nel frame precedente
        persistend_id = [self.ids[-1].index(x) for x in self.ids[-1] if x in self.ids[-2]]
        img_old_g = cv2.cvtColor(self._img_array[-2], cv2.COLOR_BGR2GRAY)
        img_now_g = cv2.cvtColor(self._img_array[-1], cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(img_now_g, img_old_g)

        _, mov_mask = cv2.threshold(frame_diff, self._mov_diff_th, 255, cv2.THRESH_BINARY)


        img_mov = cv2.bitwise_and(self._img_array[-1], self._img_array[-1], mask = mov_mask)


        for id in persistend_id:
            box = self._bb_buffer[-1][id]
            print(box)
            print(img_mov.shape)
            roi = img_mov[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # cv2.imshow("boh", roi)
            # cv2.waitKey(0)

            hoc = []
            for c in range(3):
                hoc.extend(cv2.calcHist([roi], [c], None, [10], [1, 256]))
            self._id_hoc[id] = [x/np.sum(hoc) for x in hoc]  # normalize histogram of colors

    def proximity_hoc_recovery(self, i):

        possible = [k for k, v in self._id_dict.items() if v["status"] > 0]
        ct = self._ct_buffer[-1][i]  # position of new detection

        min_cost = self._max_recovery_dist
        min_id = None
        for k in possible:
            dist = euclidean(ct, self._id_dict[k]["ct"])
            if dist < min_cost:
                min_id = k
                min_cost = dist

        hoc_dist = cosine(self._id_hoc[min_id], self.get_single_hoc(i))
        if min_id is not None and (self._id_dict[min_id]["status"] < self._max_recovery_delay or
                                                            hoc_dist < self._max_recovery_cos):
            print("recovered id: ", min_id, "last dist", min_cost)
            self._id_dict[min_id]["status"] = -1
            self._id_dict[min_id][ct] = ct
            return True, min_id
        else:
            return False, -1

    def get_single_hoc(self, i):
        img = self._id_hoc[-1]
        box = self._bb_buffer[-1][i]
        roi = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        hoc = []
        for c in range(3):
            hoc.extend(cv2.calcHist([roi], [c], None, [10], [1, 256]))
        res = [x / np.sum(hoc) for x in hoc]  # normalize histogram of colors




    def get_present_ids(self):
        return self.ids[-1]

