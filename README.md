# CNN 画像分類 サンプル

MNISTみたいなコードをベースに拡張し続けた結果のコード。
勉強用。

This project is for study.

## License

Apache License.

## dependency

* ubuntu 16.04
* tensorflow
* opencv

## テスト環境

* python3.6.6
* tensorflow-gpu 1.10
* cuda 9.2
* cudnn 7.2.1.38

## やったこと

* 車輪の再発明
* fp16 対応(中途半端)
* multi gpu
* multi label
* データサイズが大きくても学習できるようにする

## data

```
img/class1/*.jpg
img/class2/*.jpg
..
img/classN/*.jpg
```

# execute

```
ln -s /path/to/image img
cp config/v3_modoki.pyexample config/v3_modoki.py # inception v3 を 73x73の画像に合わせた
cp config-example/baseConfig.py config/baseConfig.py
vi config/v3_modoki.py
vi config/baseConfig.py # Volta
python make-test.py 70 30 #各クラス 70 の学習データと 30のテストデータ
python train.py --config config.xv3
```

このプロジェクトでは、少量の本番同様データをバリデーションデータということにしている。

# TODO

* channels_first にする。(tensorflow のtf.layerがサポートしてないため保留)
* nccl対応(現在GPU1枚なので未定)
* ps-worker構成(マシン1台なので未定)
