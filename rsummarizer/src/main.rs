use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};
use onnxruntime::{ndarray::Array2, GraphOptimizationLevel, LoggingLevel};
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use std::cmp::Ordering;

type Error = Box<dyn std::error::Error>;

static MODEL_BYTES: &[u8] = include_bytes!("../out/model.onnx");

fn main() -> Result<(), Error> {
    // Load the GPT-2 model in the ONNX format
    let env = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = env
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_model_from_memory(MODEL_BYTES)?;

    // Initialize the GPT-2 tokenizer
    let vocab_path = "out/vocab.json";
    let merges_path = "out/merges.txt";
    let tokenizer = Gpt2Tokenizer::from_file(vocab_path, merges_path, false).unwrap();

    // Tokenize the input text
    let input_text = "Hello world! This is a sample input text.";
    //  let tokens = sp.encode(input_text, None, 5, &TruncationStrategy::LongestFirst, 0);
    let encoding = tokenizer.encode(input_text, None, 5, &TruncationStrategy::LongestFirst, 0);

    // Convert the input encoding to a tensor
    let input_arr =
        Array2::from_shape_vec((1, encoding.token_ids.len()), encoding.token_ids.clone())?;
    //let mask_arr = Array2::from_shape_vec((1, encoding.mask.len()), encoding.mask.clone())?;
    let mask_arr = Array2::from_shape_vec(
        (1, encoding.mask.len()),
        encoding.mask.iter().map(|x| *x as i64).collect(),
    )?;

    let input_tensor = vec![input_arr.into_dyn(), mask_arr.into_dyn()];

    // Run inference on the model
    let output_tensors: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor)?;

    let output_tensor = &output_tensors[0];
    // Get the shape of the output tensor
    let output_shape = output_tensor.shape();

    // Get the last dimension of the output tensor
    let vocab_size = output_shape.last().unwrap();

    // Get the argmax of the output tensor along the last
    let output_ids: Vec<i64> = output_tensor
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx as i64 % *vocab_size as i64)
        .into_iter()
        .collect();
    let output_text = tokenizer.decode(&output_ids, true, true);
    println!("Output text: {}", output_text);

    Ok(())
}
