# Semi-supervised Music Tagging Transformer


## Reference

**Semi-supervised Music Tagging Transformer**, ISMIR 2021 [[paper](https://archives.ismir.net/ismir2021/paper/000096.pdf)]

-- Minz Won, Keunwoo Choi, and Xavier Serra


**TL;DR**

- We present a new state-of-the-art music tagging model using CNN front end and Transformer back end.
- It outperforms previous state-of-the art models and also suitable for long sequence modeling.
- This model can be further improved via a powerful semi-supervised pipeline, Noisy student training.
- Also, we provide new MSD splits, which are artist-level splits.

## Requirements
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## How to use pretrained models
```
import torch
from models import MusicTaggingTransformer

# Teacher model
model = MusicTaggingTransformer(conv_ndim=128, attention_ndim=256, n_seq_cls=50)
S = torch.load('../data/teacher.ckpt')
S = {k[8:]: v for k, v in S.items()}
model.load_state_dict(S)

# Student model
model = MusicTaggingTransformer(conv_ndim=128, attention_ndim=64, n_seq_cls=50)
S = torch.load('../data/compact_student.ckpt')
S = {k[8:]: v for k, v in S.items() if (k[:7] != 'teacher')}
model.load_state_dict(S)
```

## Cleaned and artist-level stratified split (CALS)
We introduce the cleaned and artist-level stratified split (CALS) for the Million Song Dataset (MSD).
Different from the previously used dataset split ([link](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split)), our proposed split is designed to avoid unintended information leak by sharing the same artists among different sets. CALS does not have shared artists in between different sets. The distribution of CALS is described below. `None` is discarded as they are unlabled songs but from the artists in the train set.

```
Train set (labeled): 163,504 songs
Valid set (labeled): 34,730 songs
Test set (labeled): 34,913 songs
Student set (unlabeled): 516,415 songs
None (unlabeled): 249,471 songs
```

## Performance Comparison
<p align = "center">
<img src = "https://imgur.com/76StTIR.png" width=400>
</p>
<p align = "center">
Performance using the conventional MSD split. 
</p>

- Music tagging transformer outperforms other previous models.

<p align = "center">
<img src = "https://imgur.com/VSfU7yR.png" width=400>
</p>
<p align = "center">
Performance using the CALS split.
</p>

- Both Short-chunk ResNet and Music tagging transformer are successfully improved through the semi-supervised pipeline.
- Knowledge distillation improved the model's performance more than the knowledge expansion.

## Citation
```
@inproceedings{won2021transformer,
  title={Semi-supervised music tagging transformer},
  author={Won, Minz and Choi, Keunwoo and Serra, Xavier},
  booktitle={Proc. of International Society for Music Information Retrieval},
  year={2021}
}

```

## License
```
======
Semi-supervised Music Tagging Transformer

Copyright (c) 2021 ByteDance. Code developed by Minz Won.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
======
sota-music-tagging-models

Copyright (c) 2020 Music Technology Group, Universitat Pompeu Fabra. Code developed by Minz Won.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
======
ViT

Copyright (c) 2020 Phil Wang
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```