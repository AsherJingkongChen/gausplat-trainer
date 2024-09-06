pub use burn::config::Config;

#[derive(Config, Debug)]
pub struct RangeOptions {
    pub start: u64,
    pub end: u64,
    pub step: u64,
}

impl RangeOptions {
    pub fn has(
        &self,
        value: u64,
    ) -> bool {
        value >= self.start && value < self.end && value % self.step == 0
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
