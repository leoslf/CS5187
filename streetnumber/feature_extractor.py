from streetnumber.nn_classifier import *

class FeatureExtractor(NNClassifier):
    def prepare_model(self):
        nn_classifier = super().prepare_model()
        return Model(inputs=nn_classifier.input, outputs=nn_classifier.get_layer("embeddings").output)

    def fit(self, *argv):
        raise NotImplementedError("Read only model")

    def predict(self, *argv, argmax=False, **kwargs):
        return super().predict(*argv, argmax=argmax, **kwargs)

