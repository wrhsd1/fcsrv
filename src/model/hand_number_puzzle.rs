use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct HandNumberPuzzlePredictor(ImageClassifierPredictor);

impl HandNumberPuzzlePredictor {
    /// Create a new instance of the HandNumberPuzzlePredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "hand_number_puzzle.onnx",
            args,
        )?))
    }
}

impl Predictor for HandNumberPuzzlePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
