use fcsrv::{model::ModelType, BootArgs};
use ort::AllocatorType;
use std::path::PathBuf;

fn main() {
    fcsrv::model::init_predictor(&BootArgs {
        debug: false,
        bind: "0.0.0.0:8000".parse().unwrap(),
        tls_cert: None,
        tls_key: None,
        api_key: None,
        multi_image_limit: 1,
        update_check: false,
        model_dir: Some(PathBuf::from("models")),
        num_threads: 4,
        allocator: AllocatorType::Arena,
    })
    .unwrap();

    let predictor = fcsrv::model::get_predictor(ModelType::PenguinsIcon).unwrap();

    let image_file = std::fs::read(
        "images/penguins-icon/0a36f4aedb149bd1aa28f26094799253f7c8228ae0cf49c4c72f6a4e76b2782f.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4)
}
