"""

"""
import torchvision
import cv2

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Object_Detector():

    def __init__(self):
        # by default use pytorch implementation
        # of faster rcnn trained on COCO dataset

        self._img = None
        self._img_tens = None
        self._choose_model = "fasterrcnn_resnet50_fpn"
        self._cat_names = COCO_INSTANCE_CATEGORY_NAMES
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self._model_used = None
        self._model = None
        self._pred_boxes = None
        self._pred_centers = None
        self._pred_class = None
        self._img_det = None

        self._threshold = 0.6

        self._rect_thick = 2
        self._text_size = .5
        self._text_thick = 1

        self.set_model()
        self._person_indexes = []



    def img_from_file(self, img_path):
        # open image from file
        self._img = cv2.imread(img_path)
        self._img_tens = self.trans(self._img)
        self._img_name = (img_path.split("/")[-1]).split(".")[0]
        self.get_detection_bb()  # perform detection of given image
        self.get_bb_centers()


    def set_model(self):
        command = "self._model" + "=torchvision.models.detection." + self._choose_model + "(pretrained=True)"
        exec(command)
        self._model.eval()

    def get_detection_bb(self):
        if self._model is not None:
            pred = self._model([self._img_tens])  # Pass the image to the model
            pred_class = [self._cat_names[i] for i in
                          list(pred[0]['labels'].numpy())]  # Get the Prediction Score
            pred_boxes = [[box[0], box[1], box[2], box[3]] for box in list(pred[0]['boxes'].detach().numpy())]  # Bounding boxes x, y, w, h?
            pred_score = list(pred[0]['scores'].detach().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x > self._threshold][-1]  # Get list of index with score greater than threshold.

            self._pred_boxes = pred_boxes[:pred_t + 1]
            self._pred_class = pred_class[:pred_t + 1]

            temp = []
            for i in range(len(self._pred_class)):
                if self._pred_class[i] == "person":
                    temp.append(i)

            self._person_bb = [pred_boxes[i] for i in temp]
        else:
            print("please first select a model, to perform detection")


    def get_bb_centers(self):
        self._pred_centers = []
        for elem in self._person_bb:
            x_c = .5*(elem[0] + elem[2])
            y_c = .5*(elem[1] + elem[3])
            self._pred_centers.append([x_c, y_c])

    # output of data
    def return_bbs(self):
        # return bounding boxes of persons given a frame
        return(self._person_bb)

    def return_bb_centers(self):
        # return centers of bounding boxes of person given a frame
        return(self._pred_centers)





