from motPapa import DetectionFileReader
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from motPapa import IoU_calc, get_bb_pp

gt_path = "../data/gt/gt.txt"
det_path = "../Output/detections.csv"

gt_reader = DetectionFileReader(gt_path, gt=True)   # frame, x, y, w, h
det_reader = DetectionFileReader(det_path)   # frame, x1, y1, x2, y2

miss_by_frame = []
false_pos_by_frame = []
distance_by_frame = []
avg_distance_by_frame = []
iou_tot_by_frame = []
iou_avg_by_frame = []

for i in range(gt_reader.n_frames()):
    print("frame: ", i + 1)

    gt_bb, gt_ct = gt_reader.get_detection(frame=i + 1)
    det_bb, det_ct = det_reader.get_detection(frame=i + 1)

    cost_matrix = distance_matrix(gt_ct, det_ct)
    ref_ind, det_ind = linear_sum_assignment(cost_matrix)

    # centers distances
    best_cost = cost_matrix[ref_ind, det_ind].sum()
    avg_distance_by_frame.append(best_cost / len(det_ind))
    distance_by_frame.append(best_cost)

    # IoUs
    temp = []
    for k in range(len(det_ind)):
        a = get_bb_pp(gt_bb[ref_ind[k]])
        b = det_bb[det_ind[k]]
        print(a, b)
        temp.append(IoU_calc(a, b))
    iou_tot_by_frame.append(np.sum(temp))
    iou_avg_by_frame.append(np.sum(temp) / len(det_ind))




    dim_diff = len(det_ind) - len(ref_ind)
    # check missed detection and false positives per frame
    if dim_diff == 0:
        false_pos_by_frame.append(0)
        miss_by_frame.append(0)
    elif dim_diff > 0:
        false_pos_by_frame.append(dim_diff)
        miss_by_frame.append(0)
    elif dim_diff < 0:
        miss_by_frame.append(-dim_diff)
        false_pos_by_frame.append(0)


# calculate final metrics
tot_miss = np.sum(miss_by_frame)
tot_fp = np.sum(false_pos_by_frame)
avg_displacement_bf = np.sum(avg_distance_by_frame) / len(avg_distance_by_frame)
avg_displacement_raw = np.sum(distance_by_frame) / (len(distance_by_frame))

print("tot miss", tot_miss)
print("tot_fp", tot_fp)
print("ave displacement", avg_displacement_raw)
plt.plot(distance_by_frame)
plt.title("avg_dist_raw")
plt.show()


plt.plot(avg_distance_by_frame)
plt.title("avg_dist_mean")
plt.xlabel("Frame")
plt.ylabel("Distance (pixels)")
plt.savefig("../Output/metrics/avg_dist_det_plot.jpeg")
plt.show()

plt.plot(iou_avg_by_frame)
plt.title("Intersection over Union by Frame")
plt.xlabel("Frame")
plt.ylabel("IoU")
plt.savefig("../Output/metrics/avg_IoU_det_plot.jpeg")
plt.show()

print(np.argmax(distance_by_frame))
print(np.sum(iou_tot_by_frame))