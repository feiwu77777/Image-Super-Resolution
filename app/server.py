import sys
import uvicorn

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

import asyncio, aiohttp

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates

templates = Jinja2Templates(directory='app/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


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

@app.route('/')
def index(request):
    index_html = path/'templates'/'index.html'
    return HTMLResponse(index_html.open().read())

@app.route('/upload', methods=['POST'])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["photo"].read())
    img = open_image(BytesIO(img_bytes))
    uploaded_name = 'app/static/user_imgs/uploaded/test.jpg'
    img.save(uploaded_name)

    img, _, _ = learn.predict(img)

    computed_name = 'app/static/user_imgs/computed/test.jpg'
    img.save(computed_name)
    return templates.TemplateResponse('index2.html', 
            {'request': request, 
             'uploaded_path': uploaded_name[4:], 
             'computed_path' : computed_name[4:]})

if __name__ == "__main__":
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")