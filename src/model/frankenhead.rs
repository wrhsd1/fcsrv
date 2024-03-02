use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct FrankenheadPredictor(ImageClassifierPredictor);

impl FrankenheadPredictor {
    /// Create a new instance of the Frankenhead
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "frankenhead.onnx",
            args,
        )?))
    }
}

impl Predictor for FrankenheadPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
