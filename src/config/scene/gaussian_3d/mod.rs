pub use gausplat_importer::scene::point;
pub use gausplat_renderer::scene::gaussian_3d::backend;

use crate::error::Error;
use gausplat_renderer::scene::gaussian_3d::{
    spherical_harmonics::SH_C, tensor, Gaussian3dScene,
};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Distribution;
use std::fmt;
use tensor::{Data, Tensor};

#[derive(Clone, PartialEq)]
pub struct Gaussian3dSceneConfig<B: backend::Backend> {
    pub colors_sh_degree: u8,
    pub device: B::Device,
    pub points: point::Points,
}

impl<B: backend::Backend> TryFrom<Gaussian3dSceneConfig<B>>
    for Gaussian3dScene<B>
{
    type Error = Error;
    fn try_from(config: Gaussian3dSceneConfig<B>) -> Result<Self, Self::Error> {
        if config.colors_sh_degree > 4 {
            return Err(Error::Config(
                "colors_sh_degree should be no more than 4".into(),
            ));
        }

        let device = config.device;
        let point_count = config.points.len();

        let (colors_rgb, positions) = config.points.into_iter().fold(
            (
                Vec::with_capacity(point_count),
                Vec::with_capacity(point_count),
            ),
            |(mut colors_rgb, mut positions), point| {
                colors_rgb.extend(point.color_rgb.map(|c| c as f32));
                positions.extend(point.position.map(|p| p as f32));
                (colors_rgb, positions)
            },
        );

        let colors_sh = {
            let colors_rgb = Tensor::from_data(
                Data::new(colors_rgb, [point_count, 1, 3].into()).convert(),
                &device,
            );
            let mut colors_sh = (colors_rgb - 0.5) / SH_C[0][0];
            let colors_sh_count = {
                let degree_add_1 = config.colors_sh_degree as usize + 1;
                degree_add_1 * degree_add_1
            };
            if colors_sh_count > 1 {
                let colors_sh_others = Tensor::zeros(
                    [point_count, colors_sh_count - 1, 3],
                    &device,
                );
                colors_sh = Tensor::cat(vec![colors_sh, colors_sh_others], 1);
            }
            colors_sh
        };

        let positions = Tensor::from_data(
            Data::new(positions, [point_count, 3].into()).convert(),
            &device,
        );

        let opacities = Tensor::full([point_count, 1], 0.1, &device);

        let rotations = Tensor::from_floats([[1.0, 0.0, 0.0, 0.0]], &device)
            .repeat(0, point_count);

        let scalings = {
            let mut sample_max = f32::EPSILON;
            let samples = rand_distr::LogNormal::new(0.0, std::f32::consts::E)
                .map_err(Error::RandomNormalDistribution)?
                .sample_iter(&mut StdRng::seed_from_u64(0x3D65))
                .take(point_count)
                .map(|mut sample| {
                    sample = sample.max(f32::EPSILON);
                    sample_max = sample_max.max(sample);
                    sample
                })
                .collect();

            let scalings = (Tensor::from_data(
                Data::new(samples, [point_count, 1].into()).convert(),
                &device,
            ) / sample_max)
                .sqrt()
                .clamp_min(f32::EPSILON)
                .repeat(1, 3);

            scalings
        };

        let mut scene = Gaussian3dScene::new();
        scene.set_colors_sh(colors_sh.set_require_grad(true));
        scene.set_opacities(opacities.set_require_grad(true));
        scene.set_positions(positions.set_require_grad(true));
        scene.set_rotations(rotations.set_require_grad(true));
        scene.set_scalings(scalings.set_require_grad(true));

        Ok(scene)
    }
}

impl<B: backend::Backend> fmt::Debug for Gaussian3dSceneConfig<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dSceneConfig")
            .field("colors_sh_degree", &self.colors_sh_degree)
            .field("points.len()", &self.points.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn scene_try_from_config_err() {
        use super::*;

        let device = Default::default();
        let colors_sh_degree = 5;
        let points = Default::default();

        let config = Gaussian3dSceneConfig::<burn::backend::NdArray> {
            colors_sh_degree,
            device,
            points,
        };

        let scene = Gaussian3dScene::<burn::backend::NdArray>::try_from(config);
        assert!(scene.is_err(), "{:?}", scene.unwrap());
    }

    #[test]
    fn scene_try_from_config_ok() {
        use super::*;

        let device = Default::default();
        let colors_sh_degree = 3;
        let points = [
            point::Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.2],
            },
            point::Point {
                color_rgb: [0.5, 1.0, 0.2],
                position: [1.0, 0.0, -0.3],
            },
        ]
        .into_iter()
        .collect();

        let config = Gaussian3dSceneConfig::<burn::backend::NdArray> {
            colors_sh_degree,
            device,
            points,
        };

        let scene = Gaussian3dScene::<burn::backend::NdArray>::try_from(config);
        assert!(scene.is_ok(), "{:?}", scene.unwrap_err());

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
