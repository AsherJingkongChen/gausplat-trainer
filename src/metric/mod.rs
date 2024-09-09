pub mod mae;
pub mod mse;
pub mod mssim;

pub use mae::*;
pub use mse::*;
pub use mssim::*;

pub use burn::tensor::{backend::Backend, Tensor};

pub trait Metric<B: Backend> {
    /// Evaluate the value against the target.
    ///
    /// ## Returns
    ///
    /// The metric value with shape `[1]`.
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1>;
}
