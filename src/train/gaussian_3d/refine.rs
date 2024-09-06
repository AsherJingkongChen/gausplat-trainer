pub use super::*;
pub use burn::tensor::Int;

#[derive(Clone, Debug)]
pub struct Refinement<B: Backend> {
    pub config: RefinementConfig,
    pub record: RefinementRecord<B>,
}

#[derive(Config, Debug)]
pub struct RefinementConfig {
    #[config(default = 5e-3)]
    pub threshold_opacity: f64,

    #[config(default = 2e-4)]
    pub threshold_position_2d_grad_norm: f64,

    #[config(default = 8e-2)]
    pub threshold_scaling: f64,

    #[config(default = "RangeOptions::new(500, 15000, 100)")]
    pub range_densification: RangeOptions,

    #[config(default = "RangeOptions::new(1000, 4000, 1000)")]
    pub range_increasing_colors_sh_degree_max: RangeOptions,
}

pub type RefinementRecord<B> = Option<RefinementState<B>>;

#[derive(Clone, Debug, Record)]
pub struct RefinementState<B: Backend> {
    pub positions_2d_grad_norm: Tensor<B, 2>,
    pub time: Tensor<B, 2>,
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
        positions_2d_grad_norm: Tensor<AB::InnerBackend, 2>,
        radii: Tensor<AB::InnerBackend, 2, Int>,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        {
            log::debug!(
                target: "gausplat_trainer::train",
                "Gaussian3dTrainer::refine > positions_2d_grad_norm {:#?}",
                positions_2d_grad_norm
                    .to_owned()
                    .mean_dim(0).to_data().to_vec::<f32>().unwrap()
            );

            #[cfg(debug_assertions)]
            log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::refine");
        }

        return self;

        let is_visible = radii.not_equal_elem(0);

        // Increasing the renderer option `colors_sh_degree_max`

        if self
            .refinement
            .config
            .range_increasing_colors_sh_degree_max
            .has(self.iteration)
        {
            let colors_sh_degree_max =
                &mut self.options_renderer.colors_sh_degree_max;
            *colors_sh_degree_max =
                colors_sh_degree_max.add(1).min(SH_DEGREE_MAX);
        }

        // Initializing the refinement states

        let refinement = self.refinement.record.get_or_insert_with(|| {
            let device = &is_visible.device();
            let shape = is_visible.dims();
            RefinementState {
                positions_2d_grad_norm: Tensor::zeros(shape, device),
                time: Tensor::zeros(shape, device),
            }
        });

        // Updating the refinement states

        refinement.positions_2d_grad_norm =
            refinement.positions_2d_grad_norm.to_owned().mask_where(
                is_visible.to_owned(),
                refinement
                    .positions_2d_grad_norm
                    .to_owned()
                    .add(positions_2d_grad_norm.to_owned()),
            );
        refinement.time = refinement.time.to_owned().mask_where(
            is_visible.to_owned(),
            refinement.time.to_owned().add_scalar(1.0),
        );

        // Densification

        if self
            .refinement
            .config
            .range_densification
            .has(self.iteration)
        {
            // Computing the means of 2D position gradient norms

            let positions_2d_grad_norm_mean = refinement
                .positions_2d_grad_norm
                .to_owned()
                .div(refinement.time.to_owned());

            // Checking the means of 2D position gradient norms and scalings

            let is_large = self
                .scene
                .scalings()
                .inner()
                .max_dim(1)
                .greater_elem(self.refinement.config.threshold_scaling);
            let is_out = positions_2d_grad_norm_mean.to_owned().greater_elem(
                self.refinement.config.threshold_position_2d_grad_norm,
            );
            let is_opaque = self
                .scene
                .opacities()
                .inner()
                .greater_elem(self.refinement.config.threshold_opacity);
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
