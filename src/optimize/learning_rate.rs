pub use burn::{config::Config, record::Record};

use std::ops::{Deref, DerefMut, Div, Mul};

/// A learning rate that can be a constant or exponentially decayed.
#[derive(Clone, Debug)]
pub struct LearningRate {
    pub decay: f64,
    pub end: f64,
    pub record: LearningRateRecord,
}

/// A learning rate that can be a constant or exponentially decayed.
#[derive(Config, Debug, PartialEq)]
pub struct LearningRateConfig {
    // The max count to update the learning rate.
    #[config(default = "0")]
    pub count: u64,

    #[config(default = "0.0")]
    pub end: f64,
    pub start: f64,
}

#[derive(Clone, Debug, Record)]
pub struct LearningRateRecord {
    pub current: f64,
}

impl LearningRate {
    pub fn update(&mut self) -> &mut Self {
        self.record.current = self.record.current.mul(self.decay).max(self.end);
        self
    }

    #[inline]
    pub fn load_record(
        &mut self,
        record: LearningRateRecord,
    ) -> &mut Self {
        self.record = record;
        self
    }

    #[inline]
    pub fn into_record(self) -> LearningRateRecord {
        self.record
    }
}

impl LearningRateConfig {
    pub fn init(&self) -> LearningRate {
        let count = if self.count == 0 {
            f64::INFINITY
        } else {
            self.count as f64
        };
        let decay = self.end.div(self.start).powf(count.recip());
        let record = LearningRateRecord {
            current: self.start,
        };

        LearningRate {
            decay,
            end: self.end,
            record,
        }
    }
}

impl Default for LearningRate {
    #[inline]
    fn default() -> Self {
        LearningRateConfig::default().init()
    }
}

impl Default for LearningRateConfig {
    #[inline]
    fn default() -> Self {
        Self::new(1e-3)
    }
}

impl Deref for LearningRate {
    type Target = f64;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.record.current
    }
}

impl DerefMut for LearningRate {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.record.current
    }
}

impl From<f64> for LearningRateConfig {
    #[inline]
    fn from(start: f64) -> Self {
        Self::new(start)
    }
}

impl From<f64> for LearningRate {
    #[inline]
    fn from(start: f64) -> Self {
        LearningRateConfig::from(start).init()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn default() {
        use super::*;

        let config = LearningRateConfig::default();
        let mut lr = config.init();

        assert_eq!(lr.record.current, config.start);
        assert_eq!(lr.decay, 1.0);
        assert_eq!(lr.end, 0.0);

        lr.update();
        lr.update();

        assert_eq!(lr.record.current, config.start);
        assert_eq!(lr.decay, 1.0);
        assert_eq!(lr.end, 0.0);
    }

    #[test]
    fn decay() {
        use super::*;

        let decay = LearningRateConfig::new(1.6e-4)
            .with_end(1.6e-6)
            .with_count(7000)
            .init()
            .decay;
        assert_eq!(decay, 0.9993423349014151);

        let decay = LearningRateConfig::new(1.6e-4)
            .with_end(1.6e-6)
            .with_count(30000)
            .init()
            .decay;
        assert_eq!(decay, 0.9998465061085267);
    }

    #[test]
    fn update_to_end() {
        use super::*;

        let mut lr = LearningRateConfig::new(1e-1)
            .with_end(1e-5)
            .with_count(5)
            .init();

        (0..5 + 2).for_each(|_| {
            lr.update();
        });

        assert_eq!(*lr, 1e-5);
    }
}
