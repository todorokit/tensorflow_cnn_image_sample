# CNN 画像分類 サンプル

MNISTみたいなコードをベースに拡張し続けた結果のコード。
勉強用。

This project is for study.

## License

Apache License.

## dependency

* python3.6
* tensorflow '1.2'

## やったこと

* 車輪の再発明

## data

```
img/class1/*.jpg
img/class2/*.jpg
..
img/classN/*.jpg
```

# execute

```
cp config.py.example config.py
vi config.py
python make-test.py 100 30
python train.py --config config
```

