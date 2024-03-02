use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct CardistancePredictor(ImagePairClassifierPredictor);

impl CardistancePredictor {
    /// Create a new instance of the CardistancePredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "cardistance.onnx",
            args,
            false,
        )?))
    }
}

impl Predictor for CardistancePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
