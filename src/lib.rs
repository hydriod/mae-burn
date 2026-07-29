//! `mae-burn` is a Rust implementation of Masked Autoencoder (MAE) built with the Burn deep learning framework.
//!
//! It provides a Vision Transformer-based encoder/decoder, patch masking, 2D sin-cos positional encoding, and a
//! simple public API centered around [`MaskedAutoencoderViTConfig`] and [`MaskedAutoencoderViT`].
//!
//! # Features
//!
//! - MAE implementation based on Burn 0.21
//! - `Masking` for splitting images into patches and randomly masking them
//! - `PadMaskToken` for restoring masked positions
//! - 2D sin-cos positional encoding
//! - Reconstruction of masked patches through an encoder/decoder pipeline
//! - Reconstructed images returned as tensors
//!
//! # Usage
//!
//! ## Initializing the model
//!
//! ```rust
//! use burn::backend::wgpu::WgpuDevice;
//! use mae_burn::MaskedAutoencoderViTConfig;
//!
//! type B = burn::backend::Wgpu;
//!
//! let device = WgpuDevice::DefaultDevice;
//! let config = MaskedAutoencoderViTConfig::default();
//! let model = config.init::<B>(&device);
//! ```
//!
//! ## Running a forward pass
//!
//! ```rust
//! use burn::Tensor;
//!
//! # use burn::backend::wgpu::WgpuDevice;
//! # use mae_burn::MaskedAutoencoderViTConfig;
//! # type B = burn::backend::Wgpu;
//! # let device = WgpuDevice::DefaultDevice;
//! # let config = MaskedAutoencoderViTConfig::default();
//! # let model = config.init::<B>(&device);
//! let input = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
//! let output = model.forward(input);
//! ```

pub mod layer;
pub mod model;

pub use model::{MaskedAutoencoderViT, MaskedAutoencoderViTConfig};
