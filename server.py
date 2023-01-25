from typing import List, Union
from models import SynthesizerTrn
import utils
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import no_grad, LongTensor
from MoeGoe import get_label, get_text, get_label_value
from scipy.io.wavfile import write
from uuid import uuid4
import os
from json import loads

app = FastAPI()

app.mount("/tts-res-static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return {"Hello": "World"}


def get_model_conf_path(name): return f"pretrained/{name}/config.json"
def get_model_path(name): return f"pretrained/{name}/model.pth"

class PretraniedModelsProfileData(BaseModel):
    text_cleaners: List[str]

class PretraniedModelsProfile(BaseModel):
    name: str
    speakers: List[str]
    data: PretraniedModelsProfileData


@app.get("/tts/pretrained-models", response_model=List[PretraniedModelsProfile])
def read_pretranied_models():
    models = os.listdir('pretrained')
    models = filter(lambda x: os.path.isdir(f"pretrained/{x}"), models)

    def read_model_profile(name: str):
        with open(get_model_conf_path(name)) as r:
            conf = loads(r.read())
        return {"name": name, **conf}
    models = map(read_model_profile, models)
    return list(models)


class TTSSpecifyModelReq(BaseModel):
    speaker_id: int
    text: str
    conf_text: str
    model_path: str


class TTSPretrainedModelReq(BaseModel):
    speaker_id: int
    text: str
    pretrained_model: str


tts_model_mem = {}


class TTSResp(BaseModel):
    url: str
    path: str


@app.post('/tts')
def tts(req: Union[TTSPretrainedModelReq, TTSSpecifyModelReq]) -> TTSResp:
    is_pretrained = isinstance(req, TTSPretrainedModelReq)
    is_specify = not is_pretrained
    hps_ms = utils.get_hparams_from_text(req.conf_text) if is_specify else utils.get_hparams_from_file(
        get_model_conf_path(req.pretrained_model))
    text = req.text
    model_mem_key = (
        req.conf_text + req.model_path) if is_specify else req.pretrained_model
    if not model_mem_key in tts_model_mem.keys():
        n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
        emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
        net_g_ms = SynthesizerTrn(
            n_symbols,
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=n_speakers,
            emotion_embedding=emotion_embedding,
            **hps_ms.model)
        _ = net_g_ms.eval()
        utils.load_checkpoint(req.model_path if is_specify else get_model_path(
            req.pretrained_model), net_g_ms)
        tts_model_mem[model_mem_key] = net_g_ms
    else:
        net_g_ms = tts_model_mem[model_mem_key]
    length_scale, text = get_label_value(text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(
        text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')
    stn_tst = get_text(text, hps_ms, cleaned=cleaned)
    filename = f"{hps_ms.speakers[req.speaker_id]}-{uuid4()}.wav"
    out_path = f'static/{filename}'
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([req.speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    write(out_path, hps_ms.data.sampling_rate, audio)
    return {"path": filename, "url": f"/tts-res-static/{filename}"}
