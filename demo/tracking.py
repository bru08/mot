"""

"""
from motPapa import DetectionFileReader, ImageReader, StuffShower
from motPapa import TrackerByDetection

detection_file_path = "../output/detections.csv"
frames_path = "../data/img1/"
det_reader = DetectionFileReader(detection_file_path)
img_reader = ImageReader(frames_path, False)  # True load everything at init, False load 1 each time
illustrator = StuffShower()
# tracker initialization
images = [img_reader.get_frame()]
init_bb, init_ct = det_reader.get_detection(frame=1)
tracker = TrackerByDetection(images, init_bb, init_ct)

for i in range(1, det_reader.n_frames()):
    print(i + 1)
    frame = img_reader.get_frame()
    bbs, ct = det_reader.get_detection(frame=i + 1)

    tracker.update(bbs, ct)
    index_conf = tracker.check_confounders()
    # matching = tracker.lap_constrained()
    print(tracker._id_dict)
    tracker.update_id()
    ids_to_print = tracker.get_present_ids()

    # illustrator.show_det_confounding(frame, bbs, index_conf)
    illustrator.show_tracking(frame, bbs, ct, ids_to_print)
    if not illustrator.show_wait(2):
        break