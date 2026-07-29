use burn::{prelude::*, tensor::Distribution::Normal};

/// Configuration for [`PadMaskToken`].
#[derive(Debug, Clone)]
pub struct PadMaskTokenConfig {
    /// Dimension of the learned mask token.
    pub token_dim: usize,
}

impl PadMaskTokenConfig {
    /// Creates a new mask-token configuration.
    ///
    /// # Arguments
    ///
    /// - `token_dim`: Dimension of the learned mask token.
    ///
    /// # Return value
    ///
    /// Returns a [`PadMaskTokenConfig`] with the requested token dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mae_burn::layer::PadMaskTokenConfig;
    ///
    /// let config = PadMaskTokenConfig::new(512);
    /// ```
    pub fn new(token_dim: usize) -> Self {
        Self { token_dim }
    }

    /// Initializes the mask-token module on the given device.
    ///
    /// # Arguments
    ///
    /// - `device`: Target device used to initialize the learned mask token.
    ///
    /// # Return value
    ///
    /// Returns a [`PadMaskToken`] module.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn::backend::wgpu::WgpuDevice;
    /// use mae_burn::layer::PadMaskTokenConfig;
    ///
    /// type B = burn::backend::Wgpu;
    ///
    /// let device = WgpuDevice::DefaultDevice;
    /// let module = PadMaskTokenConfig::new(512).init::<B>(&device);
    /// ```
    pub fn init<B: Backend>(&self, device: &B::Device) -> PadMaskToken<B> {
        let mask_token = Tensor::<B, 3>::random([1, 1, self.token_dim], Normal(0., 1.), device);
        PadMaskToken { mask_token }
    }
}

/// Learns and inserts mask tokens for missing patch positions.
#[derive(Debug, Module)]
pub struct PadMaskToken<B: Backend> {
    mask_token: Tensor<B, 3>,
}

impl<B: Backend> PadMaskToken<B> {
    /// Appends mask tokens and restores the original patch order.
    ///
    /// # Arguments
    ///
    /// - `encoded_output`: Visible patch embeddings produced by the encoder.
    /// - `ids_restore`: Restoration indices that place visible and masked patches back in order.
    ///
    /// # Return value
    ///
    /// Returns the padded patch sequence with learned mask tokens inserted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn::{backend::wgpu::WgpuDevice, tensor::TensorData};
    /// use mae_burn::layer::PadMaskTokenConfig;
    ///
    /// type B = burn::backend::wgpu::Wgpu;
    ///
    /// let device = WgpuDevice::DefaultDevice;
    /// let module = PadMaskTokenConfig::new(32).init::<B>(&device);
    /// let encoded_output = burn::tensor::Tensor::<B, 3>::from_data(
    ///     TensorData::ones::<f32, _>([1, 3, 32]),
    ///     &device,
    /// );
    /// let ids_restore = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
    ///     TensorData::new(vec![0_i64, 1, 2, 3, 4, 5], [6]),
    ///     &device,
    /// );
    /// let output = module.forward(encoded_output, ids_restore);
    /// ```
    pub fn forward(
        &self,
        encoded_output: Tensor<B, 3>,
        ids_restore: Tensor<B, 1, Int>,
    ) -> Tensor<B, 3> {
        let batch_size = encoded_output.dims()[0];
        let num_masked_patches = encoded_output.dims()[1];
        let patch_dim = encoded_output.dims()[2];
        let num_patches = ids_restore.dims()[0];

        // Create a tensor filled with the mask token
        let mask_tokens = self.mask_token.clone().expand([
            batch_size,
            num_patches - num_masked_patches,
            patch_dim,
        ]);

        // Scatter the encoded output into the positions specified by ids_restore
        let padded_output = Tensor::cat(vec![encoded_output, mask_tokens], 1);
        let indices = ids_restore
            .reshape([1, -1, 1])
            .repeat_dim(0, batch_size)
            .repeat_dim(2, patch_dim);
        let padded_output = padded_output.gather(1, indices);

        padded_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::wgpu::WgpuDevice, tensor::TensorData};

    type B = burn::backend::wgpu::Wgpu;
    const DEVICE: WgpuDevice = WgpuDevice::DefaultDevice;

    #[test]
    fn test_pad_mask_token() {
        const BATCH_SIZE: usize = 2;
        const NUM_PATCHES: usize = 3;
        const RESTORED_PATCHES: usize = 6;
        const PATCH_DIM: usize = 32;

        let config = PadMaskTokenConfig {
            token_dim: PATCH_DIM,
        };
        let pad_mask_token = config.init(&DEVICE);

        let encoded_output = Tensor::<B, 3>::random(
            [BATCH_SIZE, NUM_PATCHES, PATCH_DIM],
            Normal(0., 1.),
            &DEVICE,
        );
        let ids_restore = Tensor::<B, 1, Int>::from_data(
            TensorData::new((0..RESTORED_PATCHES as i32).collect(), [RESTORED_PATCHES]),
            &DEVICE,
        );

        let padded_output = pad_mask_token.forward(encoded_output.clone(), ids_restore.clone());

        assert_eq!(
            padded_output.dims(),
            [BATCH_SIZE, RESTORED_PATCHES, PATCH_DIM]
        );
    }
}
