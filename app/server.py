import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


export_one_pad_file_url = 'https://drive.google.com/uc?export=download&id=16QyNnQEDZl6y3uDVxCSQ_AwrF67_Jahs'
export_one_pad_file_name = 'resnet50_one_pad_87_acc_6_june.pkl'

export_two_pads_file_url = 'https://drive.google.com/uc?export=download&id=1KpYllvvVCLiXRsCck14aXn-O808ZKA8r'
export_two_pads_file_name = 'resnet50_81_acc_29_may_2019.pkl'


# classes = ['depleted', 'low', 'threshold', 'target', 'high', 'two_pad_high_high', 'two_pad_high_low', 'two_pad_low_low']
# one_pad_classes = ['depleted', 'low', 'threshold', 'target', 'high']
path = Path(__file__).parent

models_path = path / "models"

one_pad_model_path = models_path / "one_pad_model"
two_pads_model_path = models_path / "two_pads_model"

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


# Ensure the dir exists at the current path.
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name, mode=0o777)
        print('Created %s directory' % dir_name)

async def download_file(url, dest):
    if dest.exists():
        print("File already exists.")
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_one_pad_learner():
    print('Start downloading one pad model.')
    await download_file(export_one_pad_file_url, one_pad_model_path / export_one_pad_file_name)
    try:
        learn = load_learner(one_pad_model_path, export_one_pad_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


async def setup_two_pad_learner():
    print('Start downloading two pad model.')
    await download_file(export_two_pads_file_url, two_pads_model_path / export_two_pads_file_name)
    try:
        learn = load_learner(two_pads_model_path, export_two_pads_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

ensure_dir(one_pad_model_path)
ensure_dir(two_pads_model_path)

loop = asyncio.get_event_loop()

tasks = [asyncio.ensure_future(setup_one_pad_learner())]

one_pad_model = loop.run_until_complete(asyncio.gather(*tasks))[0]

tasks = [asyncio.ensure_future(setup_two_pad_learner())]

two_pad_model = loop.run_until_complete(asyncio.gather(*tasks))[0]

loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze_two_pads', methods=['POST'])
async def analyze_two_pads(request):
    print('analyze_two_pads call')
    img_data = await request.form()
    img_bytes = await(img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = two_pad_model.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


@app.route('/analyze_one_pad', methods=['POST'])
async def analyze_one_pad(request):
    print('analyze_one_pad call')
    img_data = await request.form()
    img_bytes = await(img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = one_pad_model.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
