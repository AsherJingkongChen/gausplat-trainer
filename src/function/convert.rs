pub use crate::error::Error;
pub use burn::tensor::{backend::Backend, Tensor};
pub use gausplat_importer::dataset::gaussian_3d::Image;

use gausplat_importer::function::*;

pub fn get_tensor_from_image<B: Backend>(
    image: &Image,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Error> {
    Ok(
        Tensor::from_data(image.decode_rgb()?.into_tensor_data(), device)
            .div_scalar(255.0),
    )
}

pub fn get_image_from_tensor<B: Backend>(tensor: Tensor<B, 3>) -> RgbImage {
    tensor
        .mul_scalar(255.0)
        .add_scalar(0.5)
        .clamp(0.0, 255.0)
        .into_data()
        .into_rgb_image()
}

#[cfg(test)]
mod tests {
    #[test]
    fn convert_without_loss_between_image_and_tensor() {
        use super::*;
        use burn::backend::NdArray;
        use gausplat_importer::{
            dataset::gaussian_3d::Image, image::ImageFormat,
        };

        let device = Default::default();

        let image_encoded_target = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
            0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00,
            0x00, 0x04, 0x08, 0x02, 0x00, 0x00, 0x00, 0xc9, 0x51, 0x62, 0x17,
            0x00, 0x00, 0x00, 0x4b, 0x49, 0x44, 0x41, 0x54, 0x78, 0x01, 0x01,
            0x40, 0x00, 0xbf, 0xff, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
            0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x00, 0x0f,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
            0x1b, 0x1c, 0x1d, 0x00, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24,
            0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x00, 0x2d, 0x2e,
            0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3a, 0x3b, 0x92, 0xd0, 0x06, 0xeb, 0x36, 0xd2, 0x3d, 0x2e, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];

        let mut image_encoded_value = std::io::Cursor::new(Vec::new());
        get_image_from_tensor(
            get_tensor_from_image::<NdArray<f32>>(
                &Image {
                    image_encoded: image_encoded_target.to_owned(),
                    image_file_name: Default::default(),
                    image_id: Default::default(),
                },
                &device,
            )
            .unwrap(),
        )
        .write_to(&mut image_encoded_value, ImageFormat::Png)
        .unwrap();
        let image_encoded_value = image_encoded_value.into_inner();

        assert_eq!(image_encoded_value, image_encoded_target);
    }
}
