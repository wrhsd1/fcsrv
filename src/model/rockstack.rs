use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct RockstackPredictor(ImagePairClassifierPredictor);

impl RockstackPredictor {
    /// Create a new instance of the RockstackPredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "rockstack_v2.onnx",
            args,
            true,
        )?))
    }
}

impl Predictor for RockstackPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
