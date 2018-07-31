# mlutils
**A collection of commonly used machine learning functions in Python.**
<img align='right' height='100' src='https://github.com/prasunroy/mlutils/blob/master/assets/logo.png' />

![badge](https://github.com/prasunroy/mlutils/blob/master/assets/badge_1.svg)
![badge](https://github.com/prasunroy/mlutils/blob/master/assets/badge_2.svg)

## Installation
#### Method 1: Install using native pip
```
pip install git+https://github.com/prasunroy/mlutils.git
```
#### Method 2: Install manually
```
git clone https://github.com/prasunroy/mlutils.git
cd mlutils
python setup.py install
```

## Image Data Building Functions
### Download images from [ImageNet](http://www.image-net.org)
#### Description
```
fetch_imagenet(dst, wnids=[], limit=0, verbose=True)
    Downloads images from ImageNet (http://www.image-net.org).

    Args:
        dst     : Destination directory for images.
        wnids   : A list of wnid strings of ImageNet synsets to download.
                  Defaults to an empty list.
        limit   : Maximum number of images to download from each specified
                  synset. Downloads all images if limit <= 0. Defaults to 0.
        verbose : Flag for verbose mode. Defaults to True.

    Returns:
        None.
```
#### Example
```python
from mlutils.datasets import fetch_imagenet

# download 100 images from synset butterfly (wnid=n02274259)
fetch_imagenet(dst='imagenet', wnids=['n02274259'], limit=100, verbose=True)
```

<img src='https://github.com/prasunroy/mlutils/raw/master/assets/image.png' />

### Build labeled dataset from structurally organized images
#### Description
```
build_data(src, dst, flag=1, size=(128, 128), length=10000, verbose=True)
    Builds labeled dataset from structurally organized images.

    Args:
        src     : Source directory of labeled images. It should contain all the
                  labels as sub-directories where name of each sub-directory is
                  one class label and all images inside that sub-directory are
                  instances of that class.
        dst     : Destination directory for labelmap and data files.
        flag    : Read flag. Defaults to 1.
                  >0 -- read as color image (ignores alpha channel)
                  =0 -- read as grayscale image
                  <0 -- read as original image (keeps alpha channel)
        size    : Target size of images. Defaults to (128, 128).
        length  : Maximum number of images to be written in one data file.
                  Defaults to 10000.
        verbose : Flag for verbose mode. Defaults to True.

    Returns:
        None.

    Yields:
        A labelmap.json file containing mapping of labels into numeric class
        ids and one or more .mat files containing labeled data. Each row of the
        data is one labeled sample where the first column is a numeric class id
        and the remaining columns are one dimensional representation of the
        image pixels.
```
#### Example
```python
from mlutils.datasets import build_data

# build 64x64 grayscale image data
build_data(src='imagenet', dst='data', flag=0, size=(64, 64), length=10000, verbose=True)
```

## Custom Callbacks for Keras
### Telegram
**Sends training statistics as chat messages on Telegram using Telegram API.**
```python
Telegram(auth_token, chat_id, monitor='val_acc', out_dir='.', task_id=None)
```
| Argument | Type | Default | Description |
| :------- | :--: | :-----: | :---------- |
| **auth_token** | `string`                      | | Unique authentication token to access [Telegram API](https://core.telegram.org/api). It is obtained during the creation of a [Telegram Bot](https://core.telegram.org/bots) account.   |
| **chat_id**    | `integer` <br>or<br> `string` | | Unique identifier for the target chat or username of the target channel.      |
| **monitor**    | `string`                      | `"val_acc"` | Metric to be monitored during training a model.                   |
| **out_dir**    | `string`                      | `"."`       | Output directory for plots. It will be created if does not exist. |
| **task_id**    | `integer` <br>or<br> `string` | `None`      | Unique task identifier. If not provided a random 4-digit numeric identifier will be assigned (may not be unique). |
#### Example
```python
from mlutils.callbacks import Telegram

# create a Telegram callback instance to monitor validation loss
telegram = Telegram(auth_token='123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11', chat_id='@channelusername',
                    monitor='val_loss', out_dir='output/', task_id=0)

# invoke the callback during training a model
model.fit(x, y, callbacks=[telegram])
```

## References
>[Logo](https://github.com/prasunroy/mlutils/raw/master/assets/logo.png) is obtained from [Pixabay](https://pixabay.com) made available under [Creative Commons CC0 License](https://creativecommons.org/publicdomain/zero/1.0/deed.en).

## License
MIT License

Copyright (c) 2018 Prasun Roy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<br />
<br />

**Made with** :heart: **and GitHub**
