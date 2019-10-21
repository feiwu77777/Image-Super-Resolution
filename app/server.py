from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from flask import send_from_directory
#for matrix math
import numpy as np
from werkzeug import secure_filename
#for regular expressions, saves time dealing with string data
import re
#system level operations (like loading files)
import sys
#for reading operating system data
import os
#initalize our flask app
import uvicorn

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

import asyncio, aiohttp

app = Flask(__name__)

path = Path(__file__).parent #app/

export_file_url = 'https://www.dropbox.com/s/gkjx36g2sbd0x6v/export.pkl?raw=1'
export_file_name = 'export.pkl'


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    await download_file(export_file_url, path/'models'/export_file_name)
    defaults.device = torch.device('cpu')
    learn = load_learner(path/'models', export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

def api(full_path):
    img = open_image(full_path)
    print('image opened')
    p, predict, b = learn.predict(img)
    return p

photos = UploadSet('photos', IMAGES)
saving_path = './static/user_imgs/uploaded/'
app.config['UPLOADED_PHOTOS_DEST'] = saving_path
configure_uploads(app, photos)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if (request.method == 'POST') and ('photo' in request.files):
        filename = photos.save(request.files['photo'])

    saved_path = saving_path + filename
    result = api(saved_path)
    saved_name = './static/user_imgs/computed/' + filename
    result.save(saved_name)
    return render_template("index2.html", uploaded_path = saved_path, computed_path = saved_name)

if __name__ == "__main__":
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
