"""

"""
from motPapa import DetectionFileReader, ImageReader, StuffShower
from motPapa import TrackerByDetection
import csv

detection_file_path = "../Output/detections.csv"
frames_path = "../data/img1/"
output_video_path = "../Output/video_tracking_papa.mp4"
output_data_path = "../Output/tracking/tracking.txt"
det_reader = DetectionFileReader(detection_file_path)
img_reader = ImageReader(frames_path, False)  # True load everything at init, False load 1 each time
illustrator = StuffShower()
# tracker initialization
images = [img_reader.get_frame()]
init_bb, init_ct = det_reader.get_detection(frame=1)
tracker = TrackerByDetection(images, init_bb, init_ct)

ids_to_print = tracker.get_present_ids()
tracked_positions = []
for j in range(len(ids_to_print)):
    temp = [1, ids_to_print[j]]
    temp.extend(init_ct[j])
    tracked_positions.append(temp)


for i in range(1, det_reader.n_frames()):
    print(i + 1)
    frame = img_reader.get_frame()
    bbs, ct = det_reader.get_detection(frame=i + 1)

    tracker.update(frame, bbs, ct)
    index_conf = tracker.check_confounders()
    # matching = tracker.lap_constrained()
    # print(tracker._id_dict)
    tracker.update_id()
    print(len(tracker._img_array))
    tracker.get_hoc()
    print(tracker._id_hoc)


    ids_to_print = tracker.get_present_ids()
    # preparing data to print results
    for j in range(len(ids_to_print)):
        temp = [i+1, ids_to_print[j]]
        temp.extend(ct[j])
        tracked_positions.append(temp)


    # illustrator.show_det_confounding(frame, bbs, index_conf)
    illustrator.show_tracking(frame, bbs, ct, ids_to_print)
    if not illustrator.show_wait(2):
        break


illustrator.save_video(output_video_path)

with open(output_data_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(tracked_positions)

