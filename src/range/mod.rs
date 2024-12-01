pub use burn::config::Config;

#[derive(Config, Copy, Debug, PartialEq)]
pub struct RangeOptions {
    pub start: u64,
    pub end: u64,
    pub step: u64,
}

impl RangeOptions {
    #[inline]
    pub fn default_with_step(step: u64) -> Self {
        Self {
            step,
            ..Default::default()
        }
    }

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
