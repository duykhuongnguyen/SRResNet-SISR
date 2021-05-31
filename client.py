### Module imports ###
import requests
import json
import base64
import io
from PIL import Image


### Global Variables ###
addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


### Class declarations ###


### Function declarations ###
def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(test_url, data=img, headers=headers)
    return response


if __name__ == '__main__':
    response = post_image('examples/input/lr.png')
    dictionary = json.loads(response.text)
    value = list(dictionary.values())
    byte = io.BytesIO(base64.b64decode(value[1]))
    print(byte)
    img = Image.open(byte)
    print(img.size)
    img.save('examples/output/test.png')
