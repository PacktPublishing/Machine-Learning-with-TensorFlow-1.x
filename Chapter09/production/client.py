import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from grpc.beta import implementations
from scipy.misc import imread
from datetime import datetime


class Output:
    def __init__(self, score, label):
        self.score = score
        self.label = label

    def __repr__(self):
        return "Label: %s Score: %.2f" % (self.label, self.score)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def process_image(path, label_data, top_k=3):
    start_time = datetime.now()
    img = imread(path)

    host, port = "0.0.0.0:9000".split(":")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "pet-model"
    request.model_spec.signature_name = "predict_images"

    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            img.astype(dtype=float),
            shape=img.shape, dtype=tf.float32
        )
    )

    result = stub.Predict(request, 20.)
    scores = tf.contrib.util.make_ndarray(result.outputs["scores"])[0]
    probs = softmax(scores)
    index = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)

    outputs = []
    for i in range(top_k):
        outputs.append(Output(score=float(probs[index[i]]), label=label_data[index[i]]))

    print(outputs)
    print("total time", (datetime.now() - start_time).total_seconds())
    return outputs

if __name__ == "__main__":
    label_data = [line.strip() for line in open("production/labels.txt", 'r')]
    process_image("samples_data/dog.jpg", label_data)
    process_image("samples_data/cat.jpg", label_data)
