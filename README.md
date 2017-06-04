# CNN 画像分類 サンプル

いろいろなところで紹介されている CNN画像分類 を 綺麗?に書き直した。勉強用。

This project is for my studying.

Now only train.py can use. train-mgpu.py is debugging.(only GradientDescentOptimizer can use.)

## License

Apache License.

## dependency

* python3.5
* tensorflow '1.0.1'

## やったこと

* 初期値を収束し易くした。
* configでモデルを拡張可能にした。
* Windows化
* parameter search tool が付属
* multi GPU

## data

```
img/-1 => all label is zero
img/class1/*.jpg png gif
img/class2/*.jpg png gif
..
img/classN/*.jpg png gif
```

# execute

```
cp config.py.example config.py
vi config.py
vi modelcnn.py
python make-test.py 100 30
python train.py
```

## memo

* 初期値の周りが平(周囲全部最低スコア)なら微分もなにもないので、初期値が大事。
* parameterは直感で決めるべきではないので、ツールを使う。（実験計画法未対応）
* GPUの使い道の問題でwindowsを使いたくなるが、自動再起動がかかり、ライブラリが使えないのでお勧めしない。
