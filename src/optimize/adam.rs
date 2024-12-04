//! ## Notice
//!
//! The module was adapted from the [source code of Burn v0.14.0](
//! https://github.com/tracel-ai/burn/blob/v0.14.0/crates/burn-core/src/optim/adam.rs).
//!
//! Differences between the original module and this adaptation include:
//! 1. Improved accessibility: All structs and properties being public.
//! 2. Greater flexibility and interoperability with other modules.
//! 3. Easier debugging and testing due to public visibility.
//!
//! ## License
//!
//! MIT License
//!
//! Copyright (c) 2022 Nathaniel Simard & Burn Framework Contributors

pub use burn::{
    config::Config,
    record::Record,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

/// Adam optimizer as described in the paper:
/// ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/pdf/1412.6980.pdf).
#[derive(Clone, Debug)]
pub struct Adam<AB: AutodiffBackend, const D: usize> {
    pub config: AdamConfig,
    pub record: AdamRecord<AB::InnerBackend, D>,
}

#[derive(Config, Copy, Debug, PartialEq)]
pub struct AdamConfig {
    /// The coefficient used for computing running average of gradient.
    #[config(default = "0.9")]
    pub beta_1: f64,

    /// The coefficient used for computing running average of squared gradient.
    #[config(default = "0.999")]
    pub beta_2: f64,

    /// A value added to the denominator to improve numerical stability.
    #[config(default = "1e-8")]
    pub epsilon: f64,

    /// L2 penalty.
    pub weight_decay: Option<f64>,
}

pub type AdamRecord<B, const D: usize> = Option<AdamState<B, D>>;

#[derive(Clone, Debug, Record)]
pub struct AdamState<B: Backend, const D: usize> {
    pub moment_1: Tensor<B, D>,
    pub moment_2: Tensor<B, D>,
    pub time: i32,
}

impl AdamConfig {
    /// ## Returns
    ///
    /// An optimizer that can be used to optimize a value.
    pub fn init<AB: AutodiffBackend, const D: usize>(self) -> Adam<AB, D> {
        Adam {
            config: self,
            record: None,
        }
    }
}

impl<AB: AutodiffBackend, const D: usize> Adam<AB, D> {
    /// ## Arguments
    ///
    /// * `learning_rate` - The number to multiply the gradient by.
    /// * `value` - The value to optimize.
    /// * `grad` - The gradient of the value.
    ///
    /// ## Returns
    ///
    /// The optimized value.
    pub fn update(
        &mut self,
        learning_rate: f64,
        value: Tensor<AB, D>,
        mut grad: Tensor<AB::InnerBackend, D>,
    ) -> Tensor<AB, D> {
        let value = value.inner();

        if let Some(weight_decay) = self.config.weight_decay {
            grad = grad + value.to_owned() * weight_decay;
        }

        let mut moment_1 = grad.to_owned() * (1.0 - self.config.beta_1);
        let mut moment_2 = grad.powf_scalar(2.0) * (1.0 - self.config.beta_2);
        let mut time = 1;

        if let Some(record) = &self.record {
            moment_1 = moment_1 + record.moment_1.to_owned() * self.config.beta_1;
            moment_2 = moment_2 + record.moment_2.to_owned() * self.config.beta_2;
            time += record.time;
        }

        self.record = Some(AdamState {
            moment_1: moment_1.to_owned(),
            moment_2: moment_2.to_owned(),
            time,
        });

        let moment_1_corrected = moment_1 / (1.0 - self.config.beta_1.powi(time));
        let moment_2_corrected = moment_2 / (1.0 - self.config.beta_2.powi(time));
        let grad_corrected =
            moment_1_corrected / (moment_2_corrected.sqrt() + self.config.epsilon);

        let value = value - grad_corrected * learning_rate;

        Tensor::from_inner(value).set_require_grad(true)
    }

    pub fn to_device(
        mut self,
        device: &AB::Device,
    ) -> Self {
        self.record = self.record.map(|mut record| {
            record.moment_1 = record.moment_1.to_device(device);
            record.moment_2 = record.moment_2.to_device(device);
            record
        });

        self
    }

    #[inline]
    pub fn load_record(
        &mut self,
        record: AdamRecord<AB::InnerBackend, D>,
    ) -> &mut Self {
        self.record = record;
        self
    }

    #[inline]
    pub fn into_record(self) -> AdamRecord<AB::InnerBackend, D> {
        self.record
    }
}

impl<AB: AutodiffBackend, const D: usize> Default for Adam<AB, D> {
    #[inline]
    fn default() -> Self {
        AdamConfig::default().init()
    }
}

impl Default for AdamConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        module::Param,
    };

    #[test]
    fn with_numbers() {
        let device = Default::default();

        let learning_rate = 0.01;
        let optimizer_config = AdamConfig::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(Some(0.5));
        let mut optimizer_for_weight = optimizer_config.init();
        let mut optimizer_for_bias = optimizer_config.init::<Autodiff<NdArray>, 1>();

        let weight = Param::from_data(
            [
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ],
            &device,
        );
        let bias =
            Param::from_data([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130], &device);
        let x_1 = Tensor::<Autodiff<NdArray>, 2>::from_data(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .set_require_grad(true);
        let x_2 = Tensor::<Autodiff<NdArray>, 2>::from_data(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .set_require_grad(true);

        let mut grads = x_1
            .matmul(weight.val())
            .add(bias.val().unsqueeze())
            .backward();

        let weight = weight.consume();
        let grad = weight.1.grad_remove(&mut grads).unwrap();
        let weight = Param::initialized(
            weight.0,
            optimizer_for_weight.update(learning_rate, weight.1, grad),
        );

        let bias = bias.consume();
        let grad = bias.1.grad_remove(&mut grads).unwrap();
        let bias = Param::initialized(
            bias.0,
            optimizer_for_bias.update(learning_rate, bias.1, grad),
        );

        let mut grads = x_2
            .matmul(weight.val())
            .add(bias.val().unsqueeze())
            .backward();

        let weight = weight.consume();
        let grad = weight.1.grad_remove(&mut grads).unwrap();
        let weight = Param::initialized(
            weight.0,
            optimizer_for_weight.update(learning_rate, weight.1, grad),
        );

        let bias = bias.consume();
        let grad = bias.1.grad_remove(&mut grads).unwrap();
        let bias = Param::initialized(
            bias.0,
            optimizer_for_bias.update(learning_rate, bias.1, grad),
        );

        let weights_expected = [
            [-0.340528, 0.118929, 0.384336, 0.300010, 0.066034, 0.047154],
            [
                0.057757, -0.036690, -0.386649, 0.235010, 0.175624, -0.312133,
            ],
            [
                -0.038940, 0.016306, -0.316151, 0.228410, -0.297819, 0.293047,
            ],
            [
                -0.317929, -0.239100, -0.391449, -0.318087, -0.095948, 0.142651,
            ],
            [
                0.310050, -0.235909, 0.351736, -0.192888, 0.359710, -0.050343,
            ],
            [-0.035840, -0.030203, 0.105840, 0.172110, 0.009440, 0.363346],
        ]
        .into();
        let bias_expected = [
            -0.410499, 0.068401, -0.116999, 0.097601, 0.116601, -0.006999,
        ]
        .into();

        bias.into_value()
            .into_data()
            .assert_approx_eq(&bias_expected, 5);
        weight
            .into_value()
            .into_data()
            .assert_approx_eq(&weights_expected, 2);
    }

    #[test]
    fn without_nans() {
        let device = Default::default();

        let learning_rate = 0.01;
        let optimizer_config = AdamConfig::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(Some(0.5));
        let mut optimizer_for_weight = optimizer_config.init();
        let mut optimizer_for_bias = optimizer_config.init::<Autodiff<NdArray>, 1>();

        let mut weight = Param::from_data(
            [
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ],
            &device,
        );
        let mut bias =
            Param::from_data([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130], &device);
        let x = Tensor::<Autodiff<NdArray>, 2>::from_data(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .set_require_grad(true);

        let mut grads = x
            .to_owned()
            .matmul(weight.val())
            .add(bias.val().unsqueeze())
            .backward();

        let grad = weight.grad_remove(&mut grads).unwrap();
        weight = Param::initialized(
            weight.id.to_owned(),
            optimizer_for_weight.update(learning_rate, weight.val(), grad),
        );

        let grad = bias.grad_remove(&mut grads).unwrap();
        bias = Param::initialized(
            bias.id.to_owned(),
            optimizer_for_bias.update(learning_rate, bias.val(), grad),
        );

        let mut grads = x
            .matmul(weight.val())
            .add(bias.val().unsqueeze())
            .backward();

        let grad = weight.grad_remove(&mut grads).unwrap();
        weight = Param::initialized(
            weight.id.to_owned(),
            optimizer_for_weight.update(learning_rate, weight.val(), grad),
        );

        let grad = bias.grad_remove(&mut grads).unwrap();
        _ = Param::initialized(
            bias.id.to_owned(),
            optimizer_for_bias.update(learning_rate, bias.val(), grad),
        );

        let target = true;
        let output = weight.is_nan().bool_not().all().into_scalar();
        assert_eq!(output, target);
    }
}
