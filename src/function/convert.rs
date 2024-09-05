pub use burn::tensor::{backend::Backend, Tensor};
pub use gausplat_importer::dataset::gaussian_3d::Image;

use gausplat_importer::function::*;

pub fn get_tensor_from_image<B: Backend>(
    image: &Image,
    device: &B::Device,
) -> Result<Tensor<B, 3>, gausplat_importer::error::Error> {
    Ok(
        Tensor::from_data(image.decode_rgb()?.into_tensor_data(), device)
            .div_scalar(255.0),
    )
}

pub fn get_image_from_tensor<B: Backend>(tensor: &Tensor<B, 3>) -> RgbImage {
    tensor.to_data().into_rgb_image()
}
