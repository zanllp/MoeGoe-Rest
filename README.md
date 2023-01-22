# rest api doc
http://127.0.0.1:8000/redoc
# Links
- [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
- [Pretrained models](https://github.com/CjangCjengh/TTSModels)

## Text to speech
```
TTS or VC? (t/v):t
Text to read: こんにちは。
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Speaker ID: 0
Path to save: path\to\demo.wav
Successfully saved!
```
## Voice conversion
```
TTS or VC? (t/v):v
Path of an audio file to convert:
path\to\origin.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Original speaker ID: 0
Target speaker ID: 6
Path to save: path\to\demo.wav
Successfully saved!
```
## HuBERT-VITS
```
Path of a hubert-soft model: path\to\hubert-soft.pt
Path of an audio file to convert:
path\to\origin.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Target speaker ID: 6
Path to save: path\to\demo.wav
Successfully saved!
```
## W2V2-VITS
```
Path of a w2v2 dimensional emotion model: path\to\model.onnx
TTS or VC? (t/v):t
Text to read: こんにちは。
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Speaker ID: 0
Path of an emotion reference: path\to\reference.wav
Path to save: path\to\demo.wav
Successfully saved!
```
