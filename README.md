# CNN 画像分類 サンプル

MNISTみたいなコードをベースに拡張し続けた結果のコード。
勉強用。

This project is for study.

## License

Apache License.

## dependency

* ubuntu 16.04
* python3.5
* tensorflow '1.3'

## やったこと

* 車輪の再発明
* 長所bazelを使わなくて済む。

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
cp config/xv3.pyexample config/xv3.py # inception v3 を 73x73の画像に合わせた
vi config/xv3.py
python make-test.py 70 30 #各クラス 70 の学習データと 30のテストデータ
python train.py --config config.xv3
```

このプロジェクトでは、少量の本番同様データをバリデーションデータということにしている。

# celebA 対応メモ

* crop するとネクタイ・ネックレス・帽子が見えない可能性が高い
* inferenceの時と画像切抜きが違う(73x73にしないと枚数稼げない)
* データが片寄っているので大半が-1のデータはあまり当てにならない？
* accuracyの計算がsigmoidではなく N個のsoftmax
* メモリが少ないとデータがメモリに載らない。

# 最近の活動

* multi GPU で batch_normalizationを対応させた。
* multi GPU で celeba対応した。
* linux 対応した。
* float16対応した。(GTX1080tiでは遅くなった.型変換入っているのかも？)
