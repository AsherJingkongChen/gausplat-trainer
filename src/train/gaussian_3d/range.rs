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
