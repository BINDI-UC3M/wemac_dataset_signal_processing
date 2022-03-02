# Women Emotion Multimodal Affective Computing (WEMAC) Dataset, from UC3M4Safety Database - Speech Audio Files Processing

### Introduction
This GitHub repository aims to provide the details of the preprocessing, feature and embeddings extraction performed in the speech audio recording files of the WEMAC Database. The raw audio signals cannot be provided due to privacy and ethical issues, but other audio features or embeddings can be provided upon request. Please contact the authors for further information.

### <a href="https://www..../">[WEMAC Dataset Download]</a>

#### Demo Audio File
DEMO_audio_OculusRiftS.wav is a demonstration audio file recorded in the same conditions as the original raw speech files. At 48kHz, using Oculus Rift S embedded microphone.

# Code for the preprocessing of the speech signals on BBDDLab47
Required Python libraries:
```
numpy 1.19.5
pandas 0.25.3
```
## Audio files Pre-processing
Required Python libraries:
```
librosa v0.8.1 
```
Required files: ```main_preprocess.py, audio_tools.py, unsupervised_vad.py, librosa_feature_extraction.py, opensmile_feat_extraction.py```

Code to run:
```
python main_preprocess.py 
```

## Feature & Embeddings Extraction

### librosa, eGeMAPS & ComPARE 
Required Python libraries:
```
librosa v0.8.1 
opensmile v2.2.0 
```

Code to run:
```
python main_feat_extraction.py 
```

### DeepSpectrum
Requirements: Installation of DeepSpectrum <a href="https://github.com/DeepSpectrum/DeepSpectrum/">[here]</a> <br />
Code to run:
```
python extract_deespectrum.py 
```

#### Configuration 1 (VGG19 Embeddings):
Parameters used:
```
window_size = 1 # Window size in seconds
hop_size = 1 # Hop size in seconds
net = 'resnet50' # Neural Network used
fl = 'avg_pool' # Layer from which to extract the embeddings from
```

#### Configuration 2 (ResNet50 Embeddings):
Parameters used:
```
window_size = 1 # Window size in seconds
hop_size = 1 # Hop size in seconds
net = 'vgg19' # Neural Network used
fl = 'fc2' # Layer from which to extract the embeddings from
```

### VGGish
Requirements: 
Installation of VGGish <a href="https://github.com/tensorflow/models/tree/master/research/audioset/vggish/">[here]</a> <br />
Code reference <a href="https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF">[here]</a> <br />
Required files: ```vggish_slim.py, vggish_params.py, vggish_input.py, audio_tools.py```

Code to run:
```
python main_vggish.py 
```

Parameters used:
```
hop_size = 1 # Hop size in seconds
```


## Citation

Database release:
```bibtex
@misc{miranda2022wemac,
      title={WEMAC: Women and Emotion Multi-modal Affective Computing dataset}, 
      author={Jose A. Miranda and Esther Rituerto-González and Laura Gutiérrez-Martín and Clara Luis-Mingueza and Manuel F. Canabal and Alberto Ramírez Bárcenas and Jose M. Lanza-Gutiérrez and Carmen Peláez-Moreno and Celia López-Ongil},
      year={2022},
      eprint={2203.00456},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```

## Notes
Please note that this GitHub repository is subject to changes. Contact the corresponding authors for any request on audio features or embeddings so they can be considered to be extracted and uploaded here.

## Authors
Esther Rituerto González, erituert [at] ing(dot)uc3m(dot)es <a href="https://github.com/erituert/">[GitHub]</a> <br />
Jose Ángel Miranda Calero, jmiranda [at] ing(dot)uc3m(dot)es <a href="https://github.com/JoseCalero">[GitHub]</a> <br />

## Acknowledgements 
The authors thank all the members of the UC3M4Safety for their contribution and support of the present work!
