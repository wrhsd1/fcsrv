use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct CountingPredictor(ImageClassifierPredictor);

impl CountingPredictor {
    /// Create a new instance of the CountingPredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new("counting.onnx", args)?))
    }
}

impl Predictor for CountingPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
