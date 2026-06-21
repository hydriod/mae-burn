use burn::{module::Module, prelude::*, tensor::TensorData};
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct MaskingOutput<B: Backend> {
	pub masked_patches: Tensor<B, 3>,
	pub mask: Tensor<B, 1>,
	pub visible_patch_count: usize,
}

#[derive(Clone, Debug)]
pub struct MaskingConfig {
	mask_ratio: f64,
}

impl Default for MaskingConfig {
	fn default() -> Self {
		Self { mask_ratio: 0.75 }
	}
}

impl MaskingConfig {
	pub fn new(mask_ratio: f64) -> Self {
		assert!(
			(0.0..1.0).contains(&mask_ratio),
			"mask_ratio must be in [0.0, 1.0)"
		);

		Self { mask_ratio }
	}

	pub fn init(&self) -> Masking {
		Masking::new(self.mask_ratio)
	}
}

#[derive(Clone, Debug, Module)]
pub struct Masking {
	mask_ratio: f64,
}

impl Masking {
	pub fn new(mask_ratio: f64) -> Self {
		assert!(
			(0.0..1.0).contains(&mask_ratio),
			"mask_ratio must be in [0.0, 1.0)"
		);

		Self {
			mask_ratio,
		}
	}

	pub fn forward<B: Backend>(&self, patches: Tensor<B, 3>) -> MaskingOutput<B> {
		let [_, num_patches, _] = patches.dims();
		let visible_patch_count =
			((num_patches as f64) * (1.0 - self.mask_ratio)).round() as usize;

		// 0: visible patch, 1: masked patch
		let mut mask_values = vec![1.0_f32; num_patches];
		let mut patch_indices: Vec<usize> = (0..num_patches).collect();
		let mut rng = rand::thread_rng();
		patch_indices.shuffle(&mut rng);

		for patch_idx in patch_indices.iter().take(visible_patch_count) {
			mask_values[*patch_idx] = 0.0;
		}

		let device = patches.device();
		let mask = Tensor::<B, 1>::from_data(TensorData::new(mask_values, [num_patches]), &device);
		let keep_indicator = mask.clone().neg().add_scalar(1.0).reshape([1, num_patches, 1]);
		let masked_patches = patches * keep_indicator;

		MaskingOutput {
			masked_patches,
			mask,
			visible_patch_count,
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
		let masking = MaskingConfig::new(0.5).init();
		let patches = burn::tensor::Tensor::<B, 3>::from_data(
			TensorData::ones::<f32, _>([2, 8, 4]),
			&device,
		);

		let output = masking.forward(patches);

		assert_eq!(output.mask.dims(), [8]);
		assert_eq!(output.visible_patch_count, 4);
	}
}
