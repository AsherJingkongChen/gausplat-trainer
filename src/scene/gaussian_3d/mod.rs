pub use gausplat_importer::scene::sparse_view;
pub use gausplat_renderer::scene::gaussian_3d::Backend;

use crate::error::Error;
use gausplat_renderer::{
    consts::spherical_harmonics::SH_C,
    scene::gaussian_3d::{Data, Gaussian3dScene, Param, Tensor},
};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Distribution;
use std::fmt;

#[derive(Clone, PartialEq)]
pub struct Gaussian3dSceneConfig<B: Backend> {
    pub device: B::Device,
    pub points: sparse_view::Points,
}

impl<B: Backend> TryFrom<Gaussian3dSceneConfig<B>> for Gaussian3dScene<B> {
    type Error = Error;

    fn try_from(config: Gaussian3dSceneConfig<B>) -> Result<Self, Self::Error> {
        let device = config.device;
        let point_count = config.points.len();

        let (colors_rgb, positions) = config.points.into_iter().fold(
            (
                Vec::with_capacity(point_count * 3),
                Vec::with_capacity(point_count * 3),
            ),
            |(mut colors_rgb, mut positions), point| {
                colors_rgb.extend(point.color_rgb.map(|c| c as f32));
                positions.extend(point.position.map(|p| p as f32));
                (colors_rgb, positions)
            },
        );
        let rng = rand_distr::LogNormal::new(0.0, std::f32::consts::E)
            .map_err(Error::RandomNormalDistribution)?;

        // [P, 16, 3]
        let colors_sh = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut colors_sh = Tensor::zeros([point_count, 16, 3], device);
                let colors_rgb = Tensor::from_data(
                    Data::new(
                        colors_rgb.to_owned(),
                        [point_count, 1, 3].into(),
                    )
                    .convert(),
                    device,
                );

                colors_sh = colors_sh.slice_assign(
                    [0..point_count, 0..1, 0..3],
                    (colors_rgb - 0.5) / SH_C[0][0],
                );

                Self::make_colors_sh(colors_sh)
                    .set_require_grad(is_require_grad)
            },
            device.to_owned(),
            true,
        );

        // [P, 1]
        let opacities = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                Self::make_opacities(Tensor::full(
                    [point_count, 1],
                    0.1,
                    device,
                ))
                .set_require_grad(is_require_grad)
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let positions = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                Self::make_positions(Tensor::from_data(
                    Data::new(positions.to_owned(), [point_count, 3].into())
                        .convert(),
                    device,
                ))
                .set_require_grad(is_require_grad)
            },
            device.to_owned(),
            true,
        );

        // [P, 4] (x, y, z, w)
        let rotations = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                Self::make_rotations(Tensor::from_data(
                    Data::new(
                        [0.0, 0.0, 0.0, 1.0].repeat(point_count),
                        [point_count, 4].into(),
                    )
                    .convert(),
                    device,
                ))
                .set_require_grad(is_require_grad)
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let scalings = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                println!("INIT!!!");
                let mut sample_max = f32::EPSILON;
                let samples = rng
                    .sample_iter(&mut StdRng::seed_from_u64(0x3D65))
                    .take(point_count)
                    .map(|mut sample| {
                        sample = sample.max(f32::EPSILON);
                        sample_max = sample_max.max(sample);
                        sample
                    })
                    .collect();

                Self::make_scalings(
                    Tensor::from_data(
                        Data::new(samples, [point_count, 1].into()).convert(),
                        device,
                    )
                    .div_scalar(sample_max)
                    .sqrt()
                    .clamp_min(f32::EPSILON)
                    .repeat(1, 3),
                )
                .set_require_grad(is_require_grad)
            },
            device,
            true,
        );

        Ok(Self {
            colors_sh,
            opacities,
            positions,
            rotations,
            scalings,
        })
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dSceneConfig<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dSceneConfig")
            .field("device", &self.device)
            .field("points.len()", &self.points.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scene_try_from_config_shapes() {
        use super::*;

        let device = Default::default();
        let points = vec![
            sparse_view::Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.2],
            },
            sparse_view::Point {
                color_rgb: [0.5, 1.0, 0.2],
                position: [1.0, 0.0, -0.3],
            },
        ];

        let config =
            Gaussian3dSceneConfig::<burn::backend::NdArray> { device, points };

        let scene = Gaussian3dScene::try_from(config);
        assert!(scene.is_ok(), "{}", scene.unwrap_err());

        let scene = scene.unwrap();

        let colors_sh = scene.colors_sh();
        assert_eq!(colors_sh.dims(), [2, 16, 3]);

        let opacities = scene.opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.scalings();
        assert_eq!(scalings.dims(), [2, 3]);
    }
}
