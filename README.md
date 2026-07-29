# mae-burn

`mae-burn` is a Rust implementation of Masked Autoencoder (MAE) built with the Burn deep learning framework. It provides a Vision Transformer-based encoder/decoder, patch masking, 2D sin-cos positional encoding, and a simple public API centered around `MaskedAutoencoderViTConfig` and `MaskedAutoencoderViT`.

## Features

- MAE implementation based on Burn 0.21
- `Masking` for splitting images into patches and randomly masking them
- `PadMaskToken` for restoring masked positions
- 2D sin-cos positional encoding
- Reconstruction of masked patches through an encoder/decoder pipeline
- Reconstructed images returned as tensors

## Structure

- `src/lib.rs`: public library API
- `src/model.rs`: core `MaskedAutoencoderViT` model
- `src/layer/`: helper layers for MAE
	- `mask.rs`: patch masking
	- `pad_mask_token.rs`: masked token restoration
	- `positional_encoding.rs`: 2D positional encoding

## Requirements

- Rust 2024 edition
- burn 0.21.0
- rand 0.8

Tests use the WGPU backend.

## Installation

- Install the Rust toolchain (if you don't have it):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
rustup default stable
```

On Windows, run the installer from https://rustup.rs and follow the prompts.

- Add this crate to your project's `Cargo.toml` (from crates.io):

```toml
[dependencies]
mae-burn = "0.1.0"
```

Or use a local path during development:

```toml
[dependencies]
mae-burn = { path = "../mae-burn" }
```

## Usage

### Initializing the model

```rust
use burn::backend::wgpu::WgpuDevice;
use mae_burn::MaskedAutoencoderViTConfig;

type B = burn::backend::Wgpu;

let device = WgpuDevice::DefaultDevice;
let config = MaskedAutoencoderViTConfig::default();
let model = config.init::<B>(&device);
```

### Running a forward pass

```rust
use burn::Tensor;

let input = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let output = model.forward(input);
```

## Running

Run the test suite:

```bash
cargo test
```

Check that it builds:

```bash
cargo build
```

## Notes

- The model expects 4D image tensors shaped like `[batch, channels, height, width]`.
- The default configuration uses an image size of 224x224, patch size 16, and mask ratio 0.75.
- `forward` passes the input through the encoder and decoder and returns a reconstructed image.

## License

MIT License

## Citation

```bibtex
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```