use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct KnotsCrossesCirclePredictor(ImageClassifierPredictor);

impl KnotsCrossesCirclePredictor {
    /// Create a new instance of the KnotsCrossesCirclePredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "knotsCrossesCircle.onnx",
            args,
        )?))
    }
}

impl Predictor for KnotsCrossesCirclePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
