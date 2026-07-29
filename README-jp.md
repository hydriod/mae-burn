# mae-burn

`mae-burn` は、Rust の深層学習フレームワーク Burn を使って Masked Autoencoder (MAE) を実装したリポジトリです。Vision Transformer ベースのエンコーダ/デコーダ、パッチマスキング、2 次元の sin-cos positional encodingをまとめて提供します。

## 特徴

- Burn 0.21 ベースの MAE 実装
- 画像をパッチ化してランダムにマスクする `Masking`
- マスク位置を復元する `PadMaskToken`
- 2 次元 sin-cos positional encoding
- 学習時は masked patch の再構成誤差を損失として計算
- 推論時は再構成画像を `TensorData` として取り出し可能

## 構成

- `src/lib.rs`: ライブラリの公開 API
- `src/model.rs`: MaskedAutoencoderViT 本体
- `src/layer/`: MAE 用の補助レイヤ
	- `mask.rs`: パッチのマスキング
	- `pad_mask_token.rs`: マスクトークンの復元
	- `positional_encoding.rs`: 2 次元 positional encoding

## 依存関係

- Rust 2024 edition
- burn 0.21.0
- rand 0.8

テストでは WGPU バックエンドを使います。

## インストール

- Rust ツールチェーンが未インストールの場合は rustup を使って導入してください。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
rustup default stable
```

Windows では https://rustup.rs のインストーラを実行して指示に従ってください。

- crates.io から依存関係として追加する場合（例）:

```toml
[dependencies]
mae-burn = "0.1.0"
```

開発中にローカルパスを使う場合:

```toml
[dependencies]
mae-burn = { path = "../mae-burn" }
```

## 使い方

### モデルの初期化

```rust
use mae_burn::MaskedAutoencoderViTConfig;
use burn::backend::wgpu::WgpuDevice;

type B = burn::backend::Wgpu;

let device = WgpuDevice::DefaultDevice;
let config = MaskedAutoencoderViTConfig::default();
let model = config.init::<B>(&device);
```

### フォワード実行

```rust
use burn::Tensor;

let input = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let output = model.forward(input);
```


## 実行方法

テストの実行:

```bash
cargo test
```

ビルドのみ確認する場合:

```bash
cargo build
```

## 実装メモ

- 入力画像は `[batch, channels, height, width]` の 4 次元テンソルを想定しています。
- デフォルト設定では画像サイズ 224x224、パッチサイズ 16、マスク率 0.75 です。
- `forward` はエンコードとデコードを通して再構成画像を返します。

## ライセンス

MIT License

## 引用

```bibtex
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```