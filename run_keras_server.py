from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import flask

from boto import ec2

ec2 = ec2.connect_to_region('us-east-1', profile_name='my_profile_name')

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
graph = None

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def load_infer_model():
    # load the pre-trained Keras model
    global model
    model = load_model('data/mnist.h5', custom_objects={'contrastive_loss': contrastive_loss})
    print(model.summary())

    global graph
    graph = tf.get_default_graph()

def infer_reshape(img):
    a = np.reshape(img, [-1, 28, 28])
    return a



@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        content = flask.request.get_json(force=True)

        for p in content['re-id']:
            username = p['username']
            img_a = np.asarray(p['img_a'])
            img_b = np.asarray(p['img_b'])
            print('username')
            print(username)

        # classify the input image and then initialize the list
        # of predictions to return to the client
        with graph.as_default():
            preds = model.predict([infer_reshape(img_a), infer_reshape(img_b)])

        if preds < 0.5:
            data["predictions"] = 1
        else:
            data["predictions"] = 0

        # indicate that the request was a success
        data["success"] = True

    print('prediction')
    print(data)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_infer_model()
    app.run()