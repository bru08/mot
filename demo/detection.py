"""
Perform person detection given a list of images
using faster r cnn implementation in pytorch, pretrained on coco dataset
This chops the list in two parts, and spawn two python processes to speed up the detection
"""
from motPapa import pedestrian_detector
from os import listdir
import multiprocessing
import time
import csv


def detect_fun(images, index_rg, que, proc_id):
    temp = []
    for i in range(index_rg[0], index_rg[1]):  # len(images)):  #
        print("processing: ", images[i])
        my_detector = pedestrian_detector.Object_Detector()
        my_detector.img_from_file(inp + images[i])  # import image from file and find person's bb
        bbs = my_detector.return_bbs()  # return bb in order, no identity

        for elem in bbs:
            msgb = [i+1]
            msgb.extend(elem)
            temp.append(msgb)

    que[proc_id] = temp


if __name__ == "__main__":

    inp = "../data/img1/"
    images = listdir(inp)
    images.sort()

    processes = []
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    iniz = time.time()

    workerA = multiprocessing.Process(target=detect_fun,
                                      args=(images, (0, int(len(images)/2)), shared_dict, "a"))
    workerB = multiprocessing.Process(target=detect_fun,
                                      args=(images, (int(len(images)/2), len(images)), shared_dict, "b"))

    workerA.start()
    workerB.start()

    workerA.join()
    workerB.join()
    print("time elapsed: ", time.time() - iniz)

    res = shared_dict["a"]
    res.extend(shared_dict["b"])

    f = open("../Output/detection/detections.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerows(res)
    f.close()
