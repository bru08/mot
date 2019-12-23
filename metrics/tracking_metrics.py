import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from motPapa import bb_list_center
from scipy.spatial import distance_matrix
import motmetrics as mm

gt_path = "../data/gt/gt.txt"
track_path = "../Output/tracking/tracking.txt"
track_df = pd.read_csv(track_path, header=None)
gt_df = pd.read_csv(gt_path, header=None)
last_frame = int(np.max(gt_df.iloc[:, 0]))
print(last_frame)
acc = mm.MOTAccumulator(auto_id=True)

for i in range(last_frame):
    track = track_df.loc[track_df.iloc[:, 0] == i + 1, [1, 2, 3]]
    track_ids = (track.iloc[:, 0]).values.tolist()
    track_cc = (track.iloc[:, [1, 2]]).values.tolist()

    gt = gt_df.loc[gt_df.iloc[:, 0] == i + 1, [1, 2, 3, 4, 5]]
    gt_ids = gt.iloc[:, 0]

    gt_bb = (gt.iloc[:, [1, 2, 3, 4]]).values.tolist()
    gt_cc = bb_list_center(gt_bb)

    dist_mat = distance_matrix(gt_cc, track_cc)

    acc.update(gt_ids, track_ids, dist_mat)

mh = mm.metrics.create()
summary = mh.compute(acc,
                     mm.metrics.motchallenge_metrics,# metrics=['num_frames', 'mota', 'motp'],
                     name='acc')

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)



# number of ids
max_tracked_id = np.max(track_df.iloc[:, 1])
max_gt_id = np.max(gt_df.iloc[:, 1])
print(max_tracked_id, max_gt_id)

# trajectories length
trajectories_length = np.array(track_df[1].value_counts())
mean_traj_length = np.sum(trajectories_length)/trajectories_length.shape[0]
print(mean_traj_length)



plt.plot(trajectories_length)
plt.title("Length of tracked trajectories distribution")
plt.ylabel("tracked frames for id")
plt.xlabel("Number of ids")
plt.show()

acc.events.to_csv("out_events.txt")
