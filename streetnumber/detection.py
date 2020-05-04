from streetnumber.model import *
import cv2

def steps_from_gen(generator):
    steps = generator.n // generator.batch_size
    assert steps > 0
    return steps

# Create MSER object
mser = cv2.MSER_create(_delta = 1)

def regions_to_bboxes(regions):
    bboxes = []
    for i, region in enumerate(regions):
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        bboxes.append((y, y+h, x, x+w))
    return np.array(bboxes)

def mser_region_proposals(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_copy = img.copy()

    # detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    return regions

    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # cv2.polylines(img_copy, hulls, 1, (0, 255, 0))

    # cv2.imshow('img', img_copy)
    # cv2.waitKey(0)

    # mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    # for contour in hulls:
    #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # # this is used to find only text regions, remaining are ignored
    # text_only = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("text only", text_only)

    # cv2.waitKey(0)

    # return regions

class Cropper:
    def __init__(self, img, padding):
        self.img = img

        assert len(padding) == 2
        self.padding = np.array(padding)
        # w, h
        self.img_dimensions = img.shape[:2][::-1]

    def __call__(self, bbox):
        return self.crop(bbox)

    def crop(self, bbox):
        y1, y2, x1, x2 = bbox
        x1, y1 = np.maximum([x1, y1] - self.padding, 0)
        x2, y2 = np.minimum([x2, y2] + self.padding, self.img_dimensions)
        return self.img[y1:y2, x1:x2]

class Regions:
    def __init__(self, output_shape, img):
        self.output_shape = output_shape
        self.img = img
        self.regions = mser_region_proposals(img)
        self.bboxes = regions_to_bboxes(self.regions)
        self.cropper = Cropper(img, padding = (0, 0))
        self.patches = np.array(list(map(compose(self.resize, self.cropper), self.bboxes)))

    def resize(self, patch):
        return cv2.resize(patch.astype(np.float32), self.output_shape[:2][::-1], interpolation=cv2.INTER_AREA)

    @property
    def bboxes_opencv_order(self):
        return self.bboxes[:, [2, 0]].astype("int"), self.bboxes[:, [3, 1]].astype("int")

    @property
    def bbox_centers(self):
        a, b = self.bboxes_opencv_order
        return (a + b) // 2

    @property
    def bbox_radius(self):
        """ Just the half of the length of the diagonal """
        a, b = self.bboxes_opencv_order
        return np.linalg.norm((b - a) / 2, axis = 1)



class RegionProposer:
    def __init__(self, binary_classifier, threshold):
        self.binary_classifier = binary_classifier
        self.threshold = threshold

    @property
    def output_shape(self):
        return self.binary_classifier.input_shape

    def __call__(self, img):
        return self.detect(img)

    def detect(self, img):
        regions = Regions(self.output_shape, img)
        return self.thresholding(regions)

    def thresholding(self, regions):
        probabilities = self.binary_classifier.predict_proba(regions.patches)[:, 1]
        indices = probabilities > self.threshold

        return regions.bboxes[indices], regions.patches[indices], probabilities[indices]

class DigitDetector:
    def __init__(self, binary_classifier, threshold = 0.7, perform_nms = True, nms_threshold = 0.3):
        self.threshold = threshold
        self.perform_nms = perform_nms
        self.nms_threshold = nms_threshold
        
        self.binary_classifier = binary_classifier
        self.region_proposer = RegionProposer(self.binary_classifier, threshold = self.threshold)

    def detect(self, img):
        # Thresholded 
        bboxes, patches, probabilities = self.region_proposer(img)

        # Non-Maxima Suppression
        if self.perform_nms and len(bboxes) > 0:
            bboxes, patches, probabilities = self.nms(bboxes, patches, probabilities)

        return bboxes, patches, probabilities


    def nms(self, bboxes, patches, probabilities):
        if len(bboxes) == 0:
            return bboxes, patches, probabilities

        bboxes = np.array(bboxes, dtype = "float")
        probabilities = np.array(probabilities)

        picked = []

        y1, y2, x1, x2 = bboxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(probabilities)

        while len(indices) > 0:
            last_index = len(indices) - 1
            i = indices[last_index]
            picked.append(i)

            before_last = indices[:last_index]

            # finding the largest (x, y) for the start of the bbox
            # and the smallest (x, y) for the end of the bbox
            xx1 = np.maximum(x1[i], x1[before_last])
            yy1 = np.maximum(y1[i], y1[before_last])
            xx2 = np.minimum(x2[i], x2[before_last])
            yy2 = np.minimum(y2[i], y2[before_last])

            w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

            overlap_ratios = (w * h) / area[before_last]

            # removing all indices above the threshold
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap_ratios > self.nms_threshold)[0])))

        return bboxes[picked].astype("int"), patches[picked], probabilities[picked]

class DigitBinaryClassifier(BaseModel):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, input_shape = (32, 32, 1), output_shape = (2,), **kwargs)
        self.data_generator = ImageDataGenerator(# validation_split=self.validation_split,
                )

    @property
    def metrics(self):
        return ["accuracy"]

    def flow_from_dataframe(self, dataframe, subset = None, class_mode = "categorical", directory = "preprocessed", **kwargs):
        return self.data_generator.flow_from_dataframe(dataframe = dataframe,
                                                       subset = subset,
                                                       directory = directory,
                                                       x_col = "image_id",
                                                       y_col = "label", # ["healthy", "multiple_diseases", "rust", "scab"],
                                                       # has_ext = False,
                                                       class_mode = class_mode,
                                                       color_mode = "grayscale",
                                                       # classes = ["non-digit", "digit"],
                                                       target_size = self.input_shape[:2],
                                                       # validate_filenames = True,
                                                       **kwargs)

    def fit_df(self, train_df, validation_df, **kwargs):
        train_generator = self.flow_from_dataframe(train_df, **kwargs) # , "training", **kwargs)
        validation_generator = self.flow_from_dataframe(validation_df, **kwargs) # "validation", **kwargs)


        history = self.model.fit_generator(generator = train_generator,
                                           steps_per_epoch = steps_from_gen(train_generator),
                                           validation_data = validation_generator,
                                           validation_steps = steps_from_gen(validation_generator),
                                           epochs = self.epochs,
                                           callbacks = self.callbacks,
                                           verbose = self.verbose)
                                           # batch_size = self.batch_size)
        self.save_weights()
        return history

    def evaluate_df(self, df, **kwargs):
        test_generator = self.flow_from_dataframe(df, **kwargs) # , "testing", **kwargs)
        return self.model.evaluate_generator(generator = test_generator,
                                             steps = steps_from_gen(test_generator),
                                             # callbacks = self.callbacks,
                                             verbose = self.verbose)


    def predict_df(self, df, **kwargs):
        test_generator = self.flow_from_dataframe(df, class_mode = None, batch_size = 1, **kwargs)
        return self.model.predict_generator(generator = test_generator,
                                            steps = steps_from_gen(test_generator),
                                            # callbacks = self.callbacks,
                                            verbose = self.verbose)

    @property
    def pool_size(self):
        return (2, 2)

    @property
    def kernel_size(self):
        return (2, 2)

    def conv_block(self, num_filters, inputs):
        conv_1 = Conv2D(num_filters, kernel_size = self.kernel_size, activation = "relu", padding = "valid", name = "conv_1_filter_%d" % num_filters)(inputs)
        conv_2 = Conv2D(num_filters, kernel_size = self.kernel_size, activation = "relu", name = "conv_2_filter_%d" % num_filters)(conv_1)
        max_pool_1 = MaxPooling2D(self.pool_size)(conv_2)

        return max_pool_1

    @property
    def loss(self):
        return "binary_crossentropy"

    def prepare_model(self):
        inputs = Input(self.input_shape)

        num_filters = 32

        conv = inputs

        for i in range(3):
            conv = self.conv_block(num_filters * (i + 1), conv)
        
        flatten = Flatten()(conv)

        # FC Layers
        fc_1 = Dense(1024, activation = "relu")(flatten)
        dropout_1 = Dropout(0.5)(fc_1)
        fc_2 = Dense(512, activation = "relu")(dropout_1)
        dropout_2 = Dropout(0.5)(fc_2)
        fc_3 = Dense(256, activation = "relu")(dropout_2)
        dropout_3 = Dropout(0.5)(fc_3)
        output = Dense(self.output_shape[0], activation = "softmax", name = "output")(dropout_3)

        return Model(inputs, output, name = self.name)






        



