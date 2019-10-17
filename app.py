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

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

app = Flask(__name__)

path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'
il = ImageList.from_folder(path_hr)

arch = models.resnet34
size=( 820, 1024)

data = (ImageImageList.from_folder(path_mr).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.name)
          .transform(get_transforms(), size=size, tfm_y=True)
          .databunch(bs=1).normalize(imagenet_stats, do_y=True))
data.c = 3

def load(self, file:PathLikeOrBinaryStream=None, device:torch.device=None, strict:bool=True,
            with_opt:bool=None, purge:bool=False, remove_module:bool=False):
    "Load model and optimizer state (if `with_opt`) `file` from `self.model_dir` using `device`. `file` can be file-like (file or buffer)"
    if purge: self.purge(clear_opt=ifnone(with_opt, False))
    if device is None: device = self.data.device
    elif isinstance(device, int): device = torch.device('cuda', device)
    #source = self.path/self.model_dir/f'{file}.pth' if is_pathlike(file) else file
    source = file
    state = torch.load(source, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        model_state = state['model']
        if remove_module: model_state = remove_module_load(model_state)
        get_model(self.model).load_state_dict(model_state, strict=strict)
        if ifnone(with_opt,True):
            if not hasattr(self, 'opt'): self.create_opt(defaults.lr, self.wd)
            try:    self.opt.load_state_dict(state['opt'])
            except: pass
    else:
        if with_opt: warn("Saved filed doesn't contain an optimizer state.")
        if remove_module: state = remove_module_load(state)
        get_model(self.model).load_state_dict(state, strict=strict)
    del state
    gc.collect()
    return self

Learner.load = load

learn = unet_learner(data, arch, loss_func=F.l1_loss, 
        blur=True, norm_type=NormType.Weight)

learn.load('models/2b.pth')

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
	app.run()
