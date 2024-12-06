//! Range options module.

pub use burn::config::Config;

/// The range options.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct RangeOptions {
    /// The start of the range.
    pub start: u64,
    /// The end of the range.
    pub end: u64,
    /// The step of the range.
    pub step: u64,
}

impl RangeOptions {
    /// Create a new range with the specified step.
    #[inline]
    pub fn default_with_step(step: u64) -> Self {
        Self {
            step,
            ..Default::default()
        }
    }

    /// Check if the iteration is contained in the range.
    pub fn has(
        &self,
        iteration: u64,
    ) -> bool {
        iteration >= self.start
            && iteration < self.end
            && (iteration - self.start) % self.step == 0
    }
}

impl Default for RangeOptions {
    #[inline]
    fn default() -> Self {
        RangeOptions {
            start: 0,
            end: u64::MAX,
            step: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn has() {
        use super::*;

        let range = RangeOptions::new(1, 9, 2);

        (0..11).for_each(|i| {
            let target = i % 2 != 0 && i < 9;
            let output = range.has(i);
            assert_eq!(output, target, "range.has({i})");
        });
    }
}
