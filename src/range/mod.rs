pub use burn::config::Config;

#[derive(Config, Debug, PartialEq)]
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
        assert!(!range.has(0));
        assert!(range.has(1));
        assert!(!range.has(2));
        assert!(range.has(3));
        assert!(range.has(5));
        assert!(range.has(7));
        assert!(!range.has(9));
        assert!(!range.has(10));
    }
}
