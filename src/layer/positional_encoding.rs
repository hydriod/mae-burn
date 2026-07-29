use burn::{
    nn::PositionalEncoding,
    prelude::*,
    tensor::{
        FloatDType,
        grid::{GridOptions, meshgrid},
    },
};

const MAX_TIMESCALE: f64 = 10000.;

/// Configuration for a two-dimensional sine-cosine positional encoding.
#[derive(Clone, Debug)]
pub struct PositionalEncoding2dConfig {
    embed_dim: usize,
    grid_size: [usize; 2],
    cls_token: bool,
}

/// Generate 2D sine-cosine positional embeddings.
///
/// Return a tensor of shape `(cls_token + grid_size[0] * grid_size[1], embed_dim)` containing the positional embeddings.
///
/// # Args
///    * `embed_dim`- Dimension of the embedding.
///    * `grid_size`- Tuple of (height, width) of the grid.
///    * `cls_token`- Whether to include a positional embedding for the class token.
///    * `device`- The device to create the tensor on.
fn get_2d_sincos_pos_embed<B: Backend>(
    embed_dim: usize,
    grid_size: [usize; 2],
    cls_token: bool,
    device: &B::Device,
) -> Tensor<B, 2> {
    let grid_h: Tensor<B, 1> = Tensor::arange(0..grid_size[0] as i64, device).cast(FloatDType::F32);
    let grid_w: Tensor<B, 1> = Tensor::arange(0..grid_size[1] as i64, device).cast(FloatDType::F32);

    let grid = meshgrid(&[grid_h, grid_w], GridOptions::default());

    let pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device);
    if cls_token {
        let cls_token_embed = Tensor::zeros([1, embed_dim], device);
        let pos_embed = Tensor::cat(vec![cls_token_embed, pos_embed], 0);
        return pos_embed;
    }

    pos_embed
}

fn get_2d_sincos_pos_embed_from_grid<B: Backend>(
    embed_dim: usize,
    grid: [Tensor<B, 2>; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    let grid_size = grid[0].shape();
    let embed_dim_each = embed_dim / 2;

    let [grid_h, grid_w] = grid;
    let pos_embed_h =
        get_1d_sincos_pos_embed_from_grid(embed_dim_each, grid_h.flatten(0, 1), device)
            .reshape([grid_size[0] * grid_size[1], embed_dim_each]);
    let pos_embed_w =
        get_1d_sincos_pos_embed_from_grid(embed_dim_each, grid_w.flatten(0, 1), device)
            .reshape([grid_size[0] * grid_size[1], embed_dim_each]);

    Tensor::<B, 2>::cat(vec![pos_embed_h, pos_embed_w], 1)
}

/// Generate sine-cosine positional embeddings from positions.
///
/// Return a tensor of shape `(pos.shape()[0], embed_dim)` containing the positional embeddings.
fn get_1d_sincos_pos_embed_from_grid<B: Backend>(
    embed_dim: usize,
    pos: Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(embed_dim % 2 == 0, "Embed dimension must be even");

    let embed_dim_half = embed_dim / 2;
    // \frac{1}{10000^{2i/dim}}
    let omega = Tensor::arange(0..embed_dim_half as i64, device)
        .cast(FloatDType::F32)
        .div_scalar(embed_dim_half as f64)
        .neg()
        .mul_scalar(MAX_TIMESCALE.ln())
        .exp()
        .reshape([1, -1]);
    let pos = pos.reshape([-1, 1]);

    // \frac{pos}{10000^{2i/dim}}
    let theta = pos.matmul(omega);

    let emb_sin = theta.clone().sin();
    let emb_cos = theta.cos();
    let emb = Tensor::cat(vec![emb_sin, emb_cos], 1);

    emb
}

// TODO: positional encoding のshapeを2次元にする
impl PositionalEncoding2dConfig {
    /// Creates a new positional-encoding configuration.
    ///
    /// # Arguments
    ///
    /// - `embed_dim`: Size of the embedding dimension.
    /// - `grid_size`: Height and width of the patch grid.
    /// - `cls_token`: Whether to include a class-token embedding.
    ///
    /// # Return value
    ///
    /// Returns a [`PositionalEncoding2dConfig`] for 2D sine-cosine embeddings.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mae_burn::layer::PositionalEncoding2dConfig;
    ///
    /// let config = PositionalEncoding2dConfig::new(768, [14, 14], true);
    /// ```
    pub fn new(embed_dim: usize, grid_size: [usize; 2], cls_token: bool) -> Self {
        Self {
            embed_dim,
            grid_size,
            cls_token,
        }
    }

    /// Builds the positional-encoding module on the given device.
    ///
    /// # Arguments
    ///
    /// - `device`: Target device used to allocate the positional embedding tensor.
    ///
    /// # Return value
    ///
    /// Returns a [`PositionalEncoding`] module.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn::backend::wgpu::WgpuDevice;
    /// use mae_burn::layer::PositionalEncoding2dConfig;
    ///
    /// type B = burn::backend::Wgpu;
    ///
    /// let device = WgpuDevice::DefaultDevice;
    /// let module = PositionalEncoding2dConfig::new(768, [14, 14], true).init::<B>(&device);
    /// ```
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalEncoding<B> {
        let positional_encoding =
            get_2d_sincos_pos_embed(self.embed_dim, self.grid_size, self.cls_token, device);

        PositionalEncoding {
            sinusoids: positional_encoding.unsqueeze(),
            max_sequence_size: self.grid_size[0] * self.grid_size[1]
                + if self.cls_token { 1 } else { 0 },
            max_timescale: MAX_TIMESCALE as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;

    type B = burn::backend::wgpu::Wgpu;

    #[test]
    fn test_positional_encoding_2d() {
        let device = WgpuDevice::DefaultDevice;
        const EMBED_DIM: usize = 32;
        const GRID_SIZE: [usize; 2] = [16, 16];
        const CLS_TOKEN: bool = true;

        let config = PositionalEncoding2dConfig::new(EMBED_DIM, GRID_SIZE, CLS_TOKEN);
        let positional_encoding = config.init::<B>(&device);

        assert_eq!(
            positional_encoding.sinusoids.dims(),
            [1, GRID_SIZE[0] * GRID_SIZE[1] + 1, EMBED_DIM]
        );
    }
}
