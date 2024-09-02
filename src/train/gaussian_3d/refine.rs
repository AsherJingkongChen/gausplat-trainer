pub use super::*;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(&mut self) -> &mut Self {
        self
    }
}
