# CNN 画像分類 サンプル

いろいろなところで紹介されている CNN画像分類 を 綺麗?に書き直した。勉強用。

This project is for my studying.

Now only train.py and train-mgpu.py can usable. 

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
* GPUの使い道の問題でwindowsを使いたくなるが、自動再起動がかかり、色んなライブラリが使えないのでお勧めしない。

## MEMO 2 カジュアル tensorFlow をする上での地雷。

### 用途

* 研究向けの公開が主目的であるため、応用はおざなりに見える。
* 応用もしやすくはなったけど、地雷原は多い。
* サンプルプログラムは学習はできる。推論単体は用意されていないことが多いような。
* データ取得部分が複雑で推論を作りにくい。

### ubuntu install

* Nvidia と linux カーネル開発者が仲が悪い。
* ドライバがまともに動かない。動くまでに時間が掛かる。
* GPUを外してubuntuインストール後cpu等のドライバを最新にして、その後、gpuドライバをインストールするらしい。(windowsも一緒かも)
* GPU外しても最新のPCはubuntuをインストールできないことがある。(USB インストールが問題かも)
* windows で bazel を動かすのは難しい。=> msysに依存だが、msysがまともに動かない。patchがないとか。
* 上記問題に嵌ると、pythonのみの簡単なモデルしか動かない。
* 自作PCスキルも多少必要(新しいGPUが発売されたら、前のを売り新しいのに替えたい)
* linux構築済が安定

### 自作モデルのmulti GPU

* そのままでは、逆に遅くなる。
* 構築するには、tensorflow 内部と関数に対して、それなりに理解が必要
* cpu側(通常のメモリ？) に 重み行列(W b) を置かないと共有できない。
* cpu7700K, GTX 1080tix2 でpython だとCPUがボトルネック。1.42倍ぐらいでした。
