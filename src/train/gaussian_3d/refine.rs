pub use super::*;
pub use burn::tensor::{Distribution, Int};

#[derive(Clone, Debug)]
pub struct Refinement<B: Backend> {
    pub config: RefinementConfig,
    pub record: RefinementRecord<B>,
}

#[derive(Config, Debug)]
pub struct RefinementConfig {
    #[config(default = "RangeOptions::new(100, 15000, 100)")]
    pub range_densification: RangeOptions,

    #[config(default = "RangeOptions::new(1000, 4000, 1000)")]
    pub range_increasing_colors_sh_degree_max: RangeOptions,

    #[config(default = 5e-3)]
    pub threshold_opacity: f64,

    #[config(default = 2e-4)]
    pub threshold_position_2d_grad_norm: f64,

    #[config(default = 8e-2)]
    pub threshold_scaling: f64,
}

pub type RefinementRecord<B> = Option<RefinementState<B>>;

#[derive(Clone, Debug, Record)]
pub struct RefinementState<B: Backend> {
    pub positions_2d_grad_norm: Tensor<B, 1>,
    pub time: Tensor<B, 1>,
}

impl RefinementConfig {
    pub fn init<B: Backend>(&self) -> Refinement<B> {
        Refinement {
            config: self.to_owned(),
            record: None,
        }
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(
        &mut self,
        positions_2d_grad_norm: Tensor<AB::InnerBackend, 1>,
        radii: Tensor<AB::InnerBackend, 1, Int>,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::refine");

        // Initializing the record

        let config = &self.refinement.config;
        let device = &radii.device();
        let shape = radii.dims();
        let record =
            self.refinement
                .record
                .get_or_insert_with(|| RefinementState {
                    positions_2d_grad_norm: Tensor::zeros(shape, device),
                    time: Tensor::zeros(shape, device),
                });

        // Updating the record

        let is_visible = radii.not_equal_elem(0);

        record.positions_2d_grad_norm =
            record.positions_2d_grad_norm.to_owned().mask_where(
                is_visible.to_owned(),
                record
                    .positions_2d_grad_norm
                    .to_owned()
                    .add(positions_2d_grad_norm.to_owned()),
            );
        record.time = record.time.to_owned().mask_where(
            is_visible.to_owned(),
            record.time.to_owned().add_scalar(1.0),
        );

        // Densification

        if config.range_densification.has(self.iteration) {
            // Computing the means of 2D position gradient norms

            let positions_2d_grad_norm_mean = record
                .positions_2d_grad_norm
                .to_owned()
                .div(record.time.to_owned());

            // Obtaining the points

            let points = [
                self.scene.colors_sh.val().inner(),
                self.scene.opacities.val().inner(),
                self.scene.positions.val().inner(),
                self.scene.rotations.val().inner(),
                self.scene.scalings.val().inner(),
            ];

            // Checking the points

            let is_large = self
                .scene
                .scalings()
                .inner()
                .to_owned()
                .max_dim(1)
                .greater_elem(config.threshold_scaling);
            let is_opaque = self
                .scene
                .opacities()
                .inner()
                .greater_elem(config.threshold_opacity);
            let is_out = positions_2d_grad_norm_mean
                .greater_elem(config.threshold_position_2d_grad_norm)
                .unsqueeze_dim(1);
            let is_out_and_large =
                Tensor::cat(vec![is_out.to_owned(), is_large.to_owned()], 1)
                    .all_dim(1);
            let is_in_or_small = is_out_and_large.to_owned().bool_not();
            let is_small = is_large.to_owned().bool_not();
            let args_to_clone =
                Tensor::cat(vec![is_opaque.to_owned(), is_out, is_small], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);
            let args_to_retain =
                Tensor::cat(vec![is_opaque.to_owned(), is_in_or_small], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);
            let args_to_split =
                Tensor::cat(vec![is_opaque.to_owned(), is_out_and_large], 1)
                    .all_dim(1)
                    .squeeze::<1>(1)
                    .argwhere()
                    .squeeze(1);

            // Desifying by cloning small points

            let points_cloned = points
                .to_owned()
                .map(|p| p.select(0, args_to_clone.to_owned()));

            // Densifying by splitting large points

            let mut points_splitted = points.to_owned().map(|p| {
                p.select(0, args_to_split.to_owned()).repeat_dim(0, 3)
            });

            let scalings_splitted =
                Gaussian3dScene::make_scalings(points_splitted[4].to_owned());

            points_splitted[2] = points_splitted[2].to_owned().add(
                scalings_splitted
                    .random_like(Distribution::Normal(0.0, 1.5))
                    .mul(scalings_splitted.to_owned()),
            );
            points_splitted[4] = Gaussian3dScene::make_inner_scalings(
                scalings_splitted.div_scalar(2.0),
            );
            let points_splitted = points_splitted;

            // Retaining the points that are not selected

            let points_retained =
                points.map(|p| p.select(0, args_to_retain.to_owned()));

            // Updating the points

            self.scene
                .set_inner_colors_sh(
                    Tensor::from_inner(Tensor::cat(
                        vec![
                            points_cloned[0].to_owned(),
                            points_splitted[0].to_owned(),
                            points_retained[0].to_owned(),
                        ],
                        0,
                    ))
                    .require_grad(),
                )
                .set_inner_opacities(
                    Tensor::from_inner(Tensor::cat(
                        vec![
                            points_cloned[1].to_owned(),
                            points_splitted[1].to_owned(),
                            points_retained[1].to_owned(),
                        ],
                        0,
                    ))
                    .require_grad(),
                )
                .set_inner_positions(
                    Tensor::from_inner(Tensor::cat(
                        vec![
                            points_cloned[2].to_owned(),
                            points_splitted[2].to_owned(),
                            points_retained[2].to_owned(),
                        ],
                        0,
                    ))
                    .require_grad(),
                )
                .set_inner_rotations(
                    Tensor::from_inner(Tensor::cat(
                        vec![
                            points_cloned[3].to_owned(),
                            points_splitted[3].to_owned(),
                            points_retained[3].to_owned(),
                        ],
                        0,
                    ))
                    .require_grad(),
                )
                .set_inner_scalings(
                    Tensor::from_inner(Tensor::cat(
                        vec![
                            points_cloned[4].to_owned(),
                            points_splitted[4].to_owned(),
                            points_retained[4].to_owned(),
                        ],
                        0,
                    ))
                    .require_grad(),
                );

            let point_count_selected =
                points_cloned[0].dims()[0] + points_splitted[0].dims()[0];

            #[cfg(debug_assertions)]
            log::debug!(
                target: "gausplat_trainer::train",
                "Gaussian3dTrainer::refine > densification > point_count_selected ({})",
                point_count_selected,
            );

            // Updating the optimizer records

            if let Some(record) = &mut self.optimizer_colors_sh.record {
                let zeros_selected =
                    Tensor::zeros([point_count_selected, 16 * 3], device);
                record.moment_1 = Tensor::cat(
                    vec![
                        zeros_selected.to_owned(),
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        zeros_selected,
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
            }
            if let Some(record) = &mut self.optimizer_opacities.record {
                let zeros_selected =
                    Tensor::zeros([point_count_selected, 1], device);
                record.moment_1 = Tensor::cat(
                    vec![
                        zeros_selected.to_owned(),
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        zeros_selected,
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
            }
            if let Some(record) = &mut self.optimizer_positions.record {
                let zeros_selected =
                    Tensor::zeros([point_count_selected, 3], device);
                record.moment_1 = Tensor::cat(
                    vec![
                        zeros_selected.to_owned(),
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        zeros_selected,
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
            }
            if let Some(record) = &mut self.optimizer_rotations.record {
                let zeros_selected =
                    Tensor::zeros([point_count_selected, 4], device);
                record.moment_1 = Tensor::cat(
                    vec![
                        zeros_selected.to_owned(),
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        zeros_selected,
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
            }
            if let Some(record) = &mut self.optimizer_scalings.record {
                let zeros_selected =
                    Tensor::zeros([point_count_selected, 3], device);
                record.moment_1 = Tensor::cat(
                    vec![
                        zeros_selected.to_owned(),
                        record
                            .moment_1
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
                record.moment_2 = Tensor::cat(
                    vec![
                        zeros_selected,
                        record
                            .moment_2
                            .to_owned()
                            .select(0, args_to_retain.to_owned()),
                    ],
                    0,
                );
            }

            // Resetting the record

            let shape = [self.scene.colors_sh.val().dims()[0]];
            record.positions_2d_grad_norm = Tensor::zeros(shape, device);
            record.time = Tensor::zeros(shape, device);

            #[cfg(debug_assertions)]
            log::debug!(
                target: "gausplat_trainer::train",
                "Gaussian3dTrainer::refine > densification",
            );
        }

        // Increasing the renderer option `colors_sh_degree_max`

        if config
            .range_increasing_colors_sh_degree_max
            .has(self.iteration)
        {
            let colors_sh_degree_max =
                &mut self.options_renderer.colors_sh_degree_max;
            *colors_sh_degree_max =
                colors_sh_degree_max.add(1).min(SH_DEGREE_MAX);

            #[cfg(debug_assertions)]
            log::debug!(
                target: "gausplat_trainer::train",
                "Gaussian3dTrainer::refine > increasing_colors_sh_degree_max",
            );
        }

        self
    }
}

impl Default for RefinementConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
