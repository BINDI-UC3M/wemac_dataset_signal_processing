# Women and Emotion Multimodal Affective Computing (WEMAC) dataset, from UC3M4Safety Database - Speech and Audio Processing

<a href="https://arxiv.org/abs/2203.00456"> [Paper here] </a>

## Introduction
This GitHub subrepository aims to provide the details of the processing, feature and embeddings extraction performed in the <b>speech and audio </b> files of the WEMAC Database. The raw audio signals cannot be provided due to privacy and ethical issues, but more audio features or embeddings can be provided upon request. Please contact the authors for further information.

<table>
  <tr>
    <th>Dataset</th>
    <th>DOI</th>
  </tr>
  <tr>
    <td>WEMAC: Audio features</td>
    <td>https://doi.org/10.21950/XKHCCW</td>
  </tr>
  <tr>
    <td>WEMAC: Emotional labelling</td>
    <td>https://doi.org/10.21950/RYUCLV</td>
  </tr>   
</table>


## Methodology for Speech Signals processing

#### Demo Audio File
DEMO_audio_OculusRiftS.wav is a demonstration audio file recorded in the same conditions as the original raw speech files. At 48kHz, using Oculus Rift S embedded microphone.

Required Python libraries:
```
numpy 1.19.5
pandas 0.25.3
```

_Important Note_: For each of the instalation of the toolkits described below, we used:
  1) Conda virtual environments (<a href = "https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html"> [ref] </a> for each of the toolkits to be installed
  2) Running each .py file from each toolkit having previously activated the virtual environment and from the path to the toolkit installation folder
  
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

### PASE
Requirements: 
Installation of PASE <a href="https://github.com/santi-pdp/pase">[here]</a> <br />

Code to run:
```
python main_pase_features.py 
```

### VGGish
Requirements: 
Installation of VGGish <a href="https://github.com/tensorflow/models/tree/master/research/audioset/vggish/">[here]</a> <br />
Code reference <a href="https://colab.research.google.com/drive/1E3CaPAqCai9P9QhJ3WYPNCVmrJU4lAhF">[here]</a> <br />

Code to run:
```
python main_vggish.py 
```

Parameters used:
```
hop_size = 1 # Hop size in seconds
```


## Citation

Dataset release:
```bibtex
@misc{wemac2022miranda,
  doi = {10.48550/ARXIV.2203.00456},
  url = {https://arxiv.org/abs/2203.00456},
  author = {Miranda, Jose A. and Rituerto-González, Esther and Gutiérrez-Martín, Laura and Luis-Mingueza, Clara and Canabal, Manuel F. and Bárcenas, Alberto Ramírez and Lanza-Gutiérrez, Jose M. and Peláez-Moreno, Carmen and López-Ongil, Celia},
  title = {WEMAC: Women and Emotion Multi-modal Affective Computing dataset},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```


## Notes
Please note that this GitHub repository is subject to changes. Contact the corresponding authors for any request on audio features or embeddings so they can be considered to be extracted and uploaded here.

## Authors
Esther Rituerto González, erituert [at] ing(dot)uc3m(dot)es <a href="https://github.com/erituert/">[GitHub]</a> <br />
Jose Ángel Miranda Calero, jmiranda [at] ing(dot)uc3m(dot)es <a href="https://github.com/JoseCalero">[GitHub]</a> <br />

## Acknowledgements 
The authors thank all the members of the UC3M4Safety for their contribution and support of the present work!
