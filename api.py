### Module imports ###
from flask import Flask, request, Response
import io
import jsonpickle
import numpy as np
import cv2
import base64

from inference import Inference


### Global Variables ###
app = Flask(__name__)
infer_module = Inference('pretrained_model/model_epoch100_0.32897382974624634.pt')


### Class declarations ###


### Function declarations ###
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    sr = infer_module.infer(img)

    img_byte_arr = io.BytesIO()
    sr.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # do some fancy processing here....
    encoded_string = base64.b64encode(img_byte_arr)
    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image': encoded_string.decode()}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
