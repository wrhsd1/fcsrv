use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct DicematchMatchPredictor(ImageClassifierPredictor);

impl DicematchMatchPredictor {
    /// Create a new instance of the DicematchMatchPredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new("dicematch.onnx", args)?))
    }
}

impl Predictor for DicematchMatchPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
