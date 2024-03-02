use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

#[allow(non_camel_case_types)]
pub struct BrokenJigsawbrokenjigsaw_swap(ImagePairClassifierPredictor);

impl BrokenJigsawbrokenjigsaw_swap {
    /// Create a new instance of the BrokenJigsawbrokenjigsaw_swapl
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "BrokenJigsawbrokenjigsaw_swap.onnx",
            args,
        )?))
    }
}

impl Predictor for BrokenJigsawbrokenjigsaw_swap {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
