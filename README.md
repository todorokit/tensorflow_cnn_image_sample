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
* cuda 9.0
* cudnn 7.2.1.38

## やったこと

* 車輪の再発明
* multi gpu
* multi label
* データサイズが大きくても学習できるようにする (resizeしてない)

## data

```
img/class1/*.jpg
img/class2/*.jpg
..
img/classN/*.jpg
```

※メモリに載ること。
73(width) x 73(height) x 3(RGB) x 4(float) x imageNum

# execute

```
cp -r /path/to/image img
cp config/v3_modoki.pyexample config/v3_modoki.py # inception v3 を 73x73の画像に合わせた
cp config-example/baseConfig.py config/baseConfig.py
vi config/v3_modoki.py
vi config/baseConfig.py # Volta
python make-test.py 70 30 #各クラス 70 の学習データと 30のテストデータ
python train.py --config config.xv3
```

このプロジェクトでは、少量の本番同様データをバリデーションデータということにしている。

# 最近の活動

* Titan V でfloat16 のテストした。精度がでるようにした？速度は1080tiの2.5倍程度？手放しに10倍速くなると思ったのに、調整して2.5倍。
* 画像数100万でも学習できるようにした。
* multi GPU で batch_normalizationを対応させた。
* multi GPU で celeba対応した。
* linux 対応した。
* float16対応した。(GTX1080tiでは遅くなった.型変換入っているのかも？)

# TODO
* Large Dataset (multi labelも) で shuffleできるようにする。
* tf.layers.batch_normalizationをつかう
* channels_first にする。
