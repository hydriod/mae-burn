use burn::{
    module::Module,
    prelude::*,
    tensor::{Int, TensorData},
};
use rand::seq::SliceRandom;

/// Output of [`Masking::forward`].
#[derive(Debug)]
pub struct MaskingOutput<B: Backend> {
    /// Patches that remain visible after masking.
    pub masked_patches: Tensor<B, 3>,
    /// Binary mask where masked patches are marked with `1`.
    pub mask: Tensor<B, 1>,
    /// Indices that restore the original patch ordering.
    pub ids_restore: Tensor<B, 1, Int>,
}

/// Configuration for [`Masking`].
#[derive(Clone, Debug)]
pub struct MaskingConfig {
    /// Fraction of patches to mask.
    pub mask_ratio: f64,
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self { mask_ratio: 0.75 }
    }
}

impl MaskingConfig {
    /// Creates a new masking configuration.
    pub fn new(mask_ratio: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&mask_ratio),
            "mask_ratio must be in [0.0, 1.0)"
        );

        Self { mask_ratio }
    }

    /// Builds the masking module.
    ///
    /// # Example
    /// ```
    /// let model = mae_burn::layer::MaskingConfig::default().init();
    /// ```
    pub fn init(&self) -> Masking {
        Masking::new(self.mask_ratio)
    }
}

/// Randomly masks a fraction of input patches.
#[derive(Clone, Debug, Module)]
pub struct Masking {
    /// Fraction of patches to mask.
    pub mask_ratio: f64,
}

impl Masking {
    /// Creates a new masking layer.
    pub fn new(mask_ratio: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&mask_ratio),
            "mask_ratio must be in [0.0, 1.0)"
        );

        Self { mask_ratio }
    }

    /// Applies masking to a batch of patch embeddings.
    ///
    /// # Example
    /// ```
    /// use mae_burn::layer::mask::MaskingConfig;
    /// use burn::{backend::wgpu::WgpuDevice, tensor::TensorData};
    /// type B = burn::backend::wgpu::Wgpu;
    ///
    /// fn applies_mask_ratio_to_patch_count() {
    ///     let device = WgpuDevice::DefaultDevice;
    ///     const NUM_BATCHES: usize = 2;
    ///     const NUM_PATCHES: usize = 8;
    ///     const PATCH_DIM: usize = 4;
    ///     const MASK_RATIO: f64 = 0.5;
    ///
    ///     let masking = MaskingConfig::new(MASK_RATIO).init();
    ///     let patches = burn::tensor::Tensor::<B, 3>::from_data(
    ///         TensorData::ones::<f32, _>([NUM_BATCHES, NUM_PATCHES, PATCH_DIM]),
    ///         &device,
    ///     );
    ///
    ///     let output = masking.forward(patches);
    ///
    ///     assert_eq!(output.mask.dims(), [NUM_PATCHES]);
    ///     assert_eq!(output.ids_restore.dims(), [NUM_PATCHES]);
    ///     assert_eq!(output.masked_patches.dims()[0], NUM_BATCHES);
    ///     assert_eq!(
    ///         output.masked_patches.dims()[1],
    ///         (NUM_PATCHES as f64 * (1.0 - MASK_RATIO)) as usize
    ///     );
    ///     assert_eq!(output.masked_patches.dims()[2], PATCH_DIM);
    /// }
    /// ```
    pub fn forward<B: Backend>(&self, patches: Tensor<B, 3>) -> MaskingOutput<B> {
        let device = patches.device();
        let [batch_size, num_patches, patch_dim] = patches.dims();
        let visible_patch_count = ((num_patches as f64) * (1.0 - self.mask_ratio)).round() as usize;

        let mut patch_indices: Vec<usize> = (0..num_patches).collect();
        let mut rng = rand::thread_rng();
        patch_indices.shuffle(&mut rng);

        let visible_patch_indices: Vec<i64> = patch_indices
            .iter()
            .take(visible_patch_count)
            .map(|patch_idx| *patch_idx as i64)
            .collect();

        // Create the masked_patches tensor by gathering the visible patches
        let ids_keep = Tensor::<B, 1, Int>::from_data(
            TensorData::new(visible_patch_indices, [visible_patch_count]),
            &device,
        )
        .reshape([1, visible_patch_count, 1])
        .repeat_dim(0, batch_size)
        .repeat_dim(2, patch_dim);
        let masked_patches = patches.gather(1, ids_keep);

        // Create the ids_restore tensor
        let mut ids_restore_values = vec![0_i64; num_patches];
        for (shuffle_idx, patch_idx) in patch_indices.iter().enumerate() {
            ids_restore_values[*patch_idx] = shuffle_idx as i64;
        }
        let ids_restore = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids_restore_values, [num_patches]),
            &device,
        );

        // Create the mask tensor
        let mut mask_values = vec![1.0_f32; num_patches];
        for patch_idx in patch_indices.iter().take(visible_patch_count) {
            mask_values[*patch_idx] = 0.0;
        }
        let mask = Tensor::<B, 1>::from_data(TensorData::new(mask_values, [num_patches]), &device);

        MaskingOutput {
            masked_patches,
            mask,
            ids_restore,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MaskingConfig;
    use burn::{backend::wgpu::WgpuDevice, tensor::TensorData};

    type B = burn::backend::wgpu::Wgpu;

    #[test]
    fn can_initialize_masking() {
        let _ = MaskingConfig::default().init();
    }

    #[test]
    fn applies_mask_ratio_to_patch_count() {
        let device = WgpuDevice::DefaultDevice;
        const NUM_BATCHES: usize = 2;
        const NUM_PATCHES: usize = 8;
        const PATCH_DIM: usize = 4;
        const MASK_RATIO: f64 = 0.5;

        let masking = MaskingConfig::new(MASK_RATIO).init();
        let patches = burn::tensor::Tensor::<B, 3>::from_data(
            TensorData::ones::<f32, _>([NUM_BATCHES, NUM_PATCHES, PATCH_DIM]),
            &device,
        );

        let output = masking.forward(patches);

        assert_eq!(output.mask.dims(), [NUM_PATCHES]);
        assert_eq!(output.ids_restore.dims(), [NUM_PATCHES]);
        assert_eq!(output.masked_patches.dims()[0], NUM_BATCHES);
        assert_eq!(
            output.masked_patches.dims()[1],
            (NUM_PATCHES as f64 * (1.0 - MASK_RATIO)) as usize
        );
        assert_eq!(output.masked_patches.dims()[2], PATCH_DIM);
    }
}
