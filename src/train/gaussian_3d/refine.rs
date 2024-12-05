pub use super::*;
pub use crate::range::RangeOptions;
pub use burn::tensor::{Distribution, Int};

use gausplat_renderer::scene::gaussian_3d::SH_DEGREE_MAX;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Refiner<B: Backend> {
    pub config: RefinerConfig,
    pub record: RefinerRecord<B>,
}

#[derive(Config, Copy, Debug, PartialEq)]
pub struct RefinerConfig {
    #[config(default = "RangeOptions::new(500, 15000, 100)")]
    pub range_densification: RangeOptions,

    #[config(default = "RangeOptions::new(1000, 4000, 1000)")]
    pub range_increasing_colors_sh_degree_max: RangeOptions,

    #[config(default = "1.4 / 255.0")]
    pub threshold_opacity: f64,

    #[config(default = "2e-4")]
    pub threshold_position_2d_grad_norm: f64,

    #[config(default = "6e-2")]
    pub threshold_scaling: f64,
}

pub type RefinerRecord<B> = Option<RefinerState<B>>;

#[derive(Clone, Debug, Record)]
pub struct RefinerState<B: Backend> {
    pub positions_2d_grad_norm_sum: Tensor<B, 1>,
    /// `[N] (1 ~ )`
    pub time: Tensor<B, 1>,
}

impl RefinerConfig {
    pub fn init<B: Backend>(self) -> Refiner<B> {
        Refiner {
            config: self,
            record: None,
        }
    }
}

impl<B: Backend> Refiner<B> {
    pub fn to_device(
        mut self,
        device: &B::Device,
    ) -> Self {
        self.record = self.record.map(|mut record| {
            record.positions_2d_grad_norm_sum =
                record.positions_2d_grad_norm_sum.to_device(device);
            record.time = record.time.to_device(device);
            record
        });

        self
    }

    #[inline]
    pub fn load_record(
        &mut self,
        record: RefinerRecord<B>,
    ) -> &mut Self {
        self.record = record;
        self
    }

    #[inline]
    pub fn into_record(self) -> RefinerRecord<B> {
        self.record
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(
        &mut self,
        scene: &mut Gaussian3dScene<AB>,
        grads: &mut AB::Gradients,
        output: Gaussian3dRenderOutputAutodiff<AB>,
    ) -> &mut Self {
        // NOTE: The following factors are difficult to tune.
        const FACTOR_DEVIATION: f64 = 1.0;
        const FACTOR_SCALING_HUGE: f64 = 10.0;
        const FACTOR_SPLITTING: f64 = 0.65;

        // Specifying the parameters

        let Some(positions_2d_grad_norm) =
            output.positions_2d_grad_norm_ref.grad_remove(grads)
        else {
            return self;
        };

        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(target: "gausplat::trainer::gaussian_3d::refine", "start");

        let config = &self.refiner.config;
        let device = &output.radii.device();
        let point_count = output.radii.dims()[0];
        let record = self.refiner.record.get_or_insert_with(|| RefinerState {
            positions_2d_grad_norm_sum: Tensor::zeros([point_count], device),
            time: Tensor::ones([point_count], device),
        });

        // Updating the record

        let is_visible = output.radii.not_equal_elem(0);

        record.positions_2d_grad_norm_sum =
            record.positions_2d_grad_norm_sum.to_owned().mask_where(
                is_visible.to_owned(),
                record
                    .positions_2d_grad_norm_sum
                    .to_owned()
                    .add(positions_2d_grad_norm.to_owned()),
            );
        record.time = record
            .time
            .to_owned()
            .mask_where(is_visible, record.time.to_owned().add_scalar(1.0));

        // Densification

        if config.range_densification.has(self.iteration) {
            #[cfg(all(debug_assertions, not(test)))]
            log::debug!(target: "gausplat::trainer::gaussian_3d::refine", "densification");

            // Specifying the parameters

            let points = [
                scene.colors_sh.val().inner(),
                scene.opacities.val().inner(),
                scene.positions.val().inner(),
                scene.rotations.val().inner(),
                scene.scalings.val().inner(),
            ];
            let is_points_require_grad = [true; 5];
            let positions_2d_grad_norm_mean = record
                .positions_2d_grad_norm_sum
                .to_owned()
                .div(record.time.to_owned());
            let scalings_max = scene.get_scalings().inner().to_owned().max_dim(1);

            // Checking the points

            // L
            let is_large = scalings_max
                .to_owned()
                .greater_elem(config.threshold_scaling);
            // ~H
            let is_not_huge =
                scalings_max.lower_elem(config.threshold_scaling * FACTOR_SCALING_HUGE);
            // Q
            let is_opaque = scene
                .get_opacities()
                .inner()
                .greater_elem(config.threshold_opacity);
            // ~I
            let is_out = positions_2d_grad_norm_mean
                .greater_elem(config.threshold_position_2d_grad_norm)
                .unsqueeze_dim(1);
            // ~I & L
            let is_out_and_large =
                Tensor::cat(vec![is_out.to_owned(), is_large.to_owned()], 1).all_dim(1);
            // I | ~L
            let is_in_or_small = is_out_and_large.to_owned().bool_not();
            // ~L
            let is_small = is_large.to_owned().bool_not();

            // Q & (I | ~L) & ~H
            let args_to_retain =
                Tensor::cat(vec![is_opaque.to_owned(), is_in_or_small, is_not_huge], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);
            // Q & (~I & ~L)
            let args_to_clone =
                Tensor::cat(vec![is_opaque.to_owned(), is_out, is_small], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);
            // Q & (~I & L)
            let args_to_split =
                Tensor::cat(vec![is_opaque.to_owned(), is_out_and_large], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);

            // Retaining the points that are not selected

            let points_retained = points
                .to_owned()
                .map(|p| p.select(0, args_to_retain.to_owned()));

            // Densifying by cloning small points

            let mut points_cloned = points
                .to_owned()
                .map(|p| p.select(0, args_to_clone.to_owned()));
            let scalings_cloned =
                Gaussian3dScene::make_scalings(points_cloned[4].to_owned());

            // Moving the position randomly
            points_cloned[2] = Gaussian3dScene::make_inner_positions(
                Gaussian3dScene::make_positions(points_cloned[2].to_owned()).add(
                    scalings_cloned
                        .random_like(Distribution::Normal(0.0, FACTOR_DEVIATION))
                        .mul(scalings_cloned.to_owned()),
                ),
            );

            // Densifying by splitting large points

            let mut points_splitted =
                points.map(|p| p.select(0, args_to_split.to_owned()).repeat_dim(0, 2));
            let scalings_splitted =
                Gaussian3dScene::make_scalings(points_splitted[4].to_owned());

            // Decreasing the opacity
            points_splitted[1] = Gaussian3dScene::make_inner_opacities(
                Gaussian3dScene::make_opacities(points_splitted[1].to_owned())
                    .mul_scalar(FACTOR_SPLITTING),
            );
            // Moving the position randomly
            points_splitted[2] = Gaussian3dScene::make_inner_positions(
                Gaussian3dScene::make_positions(points_splitted[2].to_owned()).add(
                    scalings_splitted
                        .random_like(Distribution::Normal(0.0, FACTOR_DEVIATION))
                        .mul(scalings_splitted.to_owned()),
                ),
            );
            // Decreasing the scaling
            points_splitted[4] = Gaussian3dScene::make_inner_scalings(
                scalings_splitted.mul_scalar(FACTOR_SPLITTING),
            );

            // Updating the points

            let make_points = |param_index: usize| {
                Tensor::from_inner(Tensor::cat(
                    vec![
                        points_retained[param_index].to_owned(),
                        points_cloned[param_index].to_owned(),
                        points_splitted[param_index].to_owned(),
                    ],
                    0,
                ))
                .set_require_grad(is_points_require_grad[param_index])
            };

            scene
                .set_inner_colors_sh(make_points(0))
                .set_inner_opacities(make_points(1))
                .set_inner_positions(make_points(2))
                .set_inner_rotations(make_points(3))
                .set_inner_scalings(make_points(4));

            let point_count_retained = points_retained[0].dims()[0];
            let point_count_cloned = points_cloned[0].dims()[0];
            let point_count_splitted = points_splitted[0].dims()[0];
            let point_count_selected = point_count_cloned + point_count_splitted;
            let point_count_new = point_count_retained + point_count_selected;

            #[cfg(all(debug_assertions, not(test)))]
            log::debug!(
                target: "gausplat::trainer::gaussian_3d::refine",
                "densification > point_count ({}) -> ({}) = ({}R + {}C + {}S)",
                point_count, point_count_new,
                point_count_retained, point_count_cloned, point_count_splitted,
            );

            // Updating the optimizer records

            let update_optimizer = |optimizer: &mut Adam<AB, 2>| {
                let Some(record) = &mut optimizer.record else {
                    return;
                };
                let feature_count = record.moment_1.dims()[1];

                record.moment_1 = Tensor::cat(
                    vec![
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                        Tensor::zeros([point_count_selected, feature_count], device),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                        Tensor::zeros([point_count_selected, feature_count], device),
                    ],
                    0,
                );
            };

            update_optimizer(&mut self.optimizer_colors_sh);
            update_optimizer(&mut self.optimizer_opacities);
            update_optimizer(&mut self.optimizer_positions);
            update_optimizer(&mut self.optimizer_rotations);
            update_optimizer(&mut self.optimizer_scalings);

            // Resetting the record

            record.positions_2d_grad_norm_sum = Tensor::zeros([point_count_new], device);
            record.time = Tensor::ones([point_count_new], device);
        }

        // Increasing the render option `colors_sh_degree_max`

        if config
            .range_increasing_colors_sh_degree_max
            .has(self.iteration)
        {
            let colors_sh_degree_max = &mut self.options_renderer.colors_sh_degree_max;
            *colors_sh_degree_max = colors_sh_degree_max.add(1).min(SH_DEGREE_MAX);

            #[cfg(all(debug_assertions, not(test)))]
            log::debug!(
                target: "gausplat::trainer::gaussian_3d::refine",
                "increasing_colors_sh_degree_max ({})",
                colors_sh_degree_max,
            );
        }

        self
    }
}

impl<AB: AutodiffBackend> Default for Refiner<AB> {
    #[inline]
    fn default() -> Self {
        RefinerConfig::default().init()
    }
}

impl Default for RefinerConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
