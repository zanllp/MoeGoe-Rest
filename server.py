from typing import Union
from models import SynthesizerTrn
import utils
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import no_grad, LongTensor
from MoeGoe import get_label, get_text, get_label_value
from scipy.io.wavfile import write
from uuid import uuid4

app = FastAPI()

app.mount("/tts-res-static", StaticFiles(directory="static"), name="static")

@app.get("/")

def read_root():
    return {"Hello": "World"}



class TTSReq(BaseModel):
    conf_text: str
    speaker_id: int
    text: str
    model_path: str

tts_model_mem = {}

@app.post('/tts')
def tts(req: TTSReq):
    hps_ms = utils.get_hparams_from_text(req.conf_text)
    model_mem_key = (req.conf_text + req.model_path)
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
        utils.load_checkpoint(req.model_path, net_g_ms)
        tts_model_mem[model_mem_key] = net_g_ms
    else:
        net_g_ms = tts_model_mem[model_mem_key]
    text = req.text
    length_scale, text = get_label_value(text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')
    stn_tst = get_text(text, hps_ms, cleaned=cleaned)
    out_path = f'static/{hps_ms.speakers[req.speaker_id]}-{uuid4()}.wav'
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([req.speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    write(out_path, hps_ms.data.sampling_rate, audio)
    return { "path": out_path.split('/')[1] }

