# CNN 画像分類 サンプル

いろいろなところで紹介されている CNN画像分類 を 綺麗?に書き直した。
改良が加わり、画像サイズ固定や貧弱なGPU(環境)用にカスタマイズ可能なモデル作成するためのライブラリ兼ツールとなった。
This repository use for making model that image size fixed or poor gpu enviroment.

## License
Apache License.

## requirement
python3.5
tensorflow '1.0.1'

## やったこと

* 初期値を収束し易くした。
* configでモデルを拡張可能にした。
* Windows化
* parameter search tool が付属

# memo

* 初期値が悪ければ微分もなにもないので、初期値が大事。
* parameterは直感で決めるべきではないので、ツールを使う。（実験計画法未対応）

