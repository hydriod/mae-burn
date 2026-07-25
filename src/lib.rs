//! Masked autoencoder components built on `burn`.

pub mod layer;
pub mod model;

pub use model::{MaskedAutoencoderViT, MaskedAutoencoderViTConfig};
