pub use super::*;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn get_loss_colors_rgb_2d(
        &self,
        value: Tensor<AB, 3>,
        target: Tensor<AB, 3>,
    ) -> Tensor<AB, 1> {
        const RANGE_STEP_OPTIMIZATION_FINE: u64 = 2;

        if self.iteration % RANGE_STEP_OPTIMIZATION_FINE == 0 {
            self.metric_optimization_coarse.evaluate(value, target)
        } else {
            self.metric_optimization_coarse
                .evaluate(value.to_owned(), target.to_owned())
                .add(
                    self.metric_optimization_fine
                        .evaluate(value.movedim(2, 0), target.movedim(2, 0)),
                )
                .div_scalar(2.0)
        }
    }
}
