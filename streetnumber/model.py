import os

from datetime import datetime
from streetnumber.utils import *

class BaseModel:
    def __init__(self,
                 input_shape = (9, ),
                 output_shape = (3, ),
                 batch_size = 10000,
                 epochs = 1000,
                 verbose = 2,
                 use_multiprocessing = False,
                 compiled = False,
                 *argv, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.compiled = compiled
        self.epochs = epochs
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.__dict__.update(kwargs)

        self.init()
        self.model = self.prepare_model()

        try:
            self.load_weights()
        except:
            raise ImportError("Could not load pretrained model weights")

        if not self.compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            print ("compiled: %s" % self.__class__.__name__)

        self.model.summary()

    @property
    def name(self):
        return self.__class__.__name__

    def init(self):
        pass

    @property
    def optimizer(self):
        return "adam"

    @property
    def loss(self):
        return "mean_squared_error"

    @property
    def weight_filename(self):
        return "%s.h5" % self.name

    def load_weights(self, filename = None):
        if filename is None:
            filename = self.weight_filename

        if os.path.exists(filename):
            self.model.load_weights(filename, by_name=True)

    def save_weights(self):
        self.model.save_weights(self.weight_filename)

    @property
    def metrics(self):
        return ["accuracy", recall, precision, F1, "categorical_crossentropy"]

    @property
    def earlystopping(self):
       return EarlyStopping(monitor="val_loss", # use validation accuracy for stopping
                            min_delta = 0.0001,
                            patience = 50, 
                            verbose = self.verbose,
                            mode="auto")

    @property
    def modelcheckpoint(self):
        return ModelCheckpoint(os.path.join(self.logdir, "epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), monitor="val_loss", save_weights_only=True, save_best_only=True, period=3)


    @property
    def callbacks(self):
        return [
            self.earlystopping,
            self.modelcheckpoint,
            TensorBoard(log_dir=self.logdir),
        ]

    @property
    def logdir(self):
        return "logs/%s/%s" % (self.__class__.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def prepare_model(self):
        return None

    def fit(self, train_X, train_Y, validation = None):
        history = self.model.fit(train_X, train_Y,
                                 validation_data = validation,
                                 validation_split = 0.3,
                                 batch_size = self.batch_size,
                                 epochs = self.epochs,
                                 callbacks = self.callbacks,
                                 verbose = self.verbose,
                                 use_multiprocessing = self.use_multiprocessing)
        self.save_weights()
        return history

    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def predict(self, X, *argv, **kwargs):
        return self.model.predict_class(X, *argv, **kwargs)

