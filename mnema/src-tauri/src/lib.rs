#![allow(non_local_definitions)]

use base64::Engine;
use chrono::Local;
use image::{GenericImageView, ImageOutputFormat};
use screenshots::Screen;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{Emitter, Manager};

// --- AI & Math Imports ---
use ndarray::{Array2, Array3, Array4, ArrayD, Axis, Ix3};
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::{Tensor, Value};
use sentencepiece::SentencePieceProcessor;
use std::ops::Deref;
use std::sync::Mutex;

// --- DATABASE IMPORTS ---
use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::Table;
use std::sync::Arc;

// --- GLOBAL STATE STRUCT ---
pub struct AppState {
    pub db_table: Option<Table>,
    pub text_session: Option<Session>,
    pub ocr_state: Option<OcrState>,
}

#[derive(Clone)]
pub struct OcrCacheEntry {
    pub timestamp: String,
    pub text: String,
    pub links: Vec<String>,
}

pub struct OcrState {
    pub encoder_session: Session,
    pub decoder_init_session: Session,
    pub decoder_with_past_session: Session,
    pub tokenizer: SentencePieceProcessor,
    pub cache: HashMap<String, OcrCacheEntry>,
}

#[derive(serde::Serialize, Clone)]
pub struct OcrResult {
    pub id: String,
    pub timestamp: String,
    pub text: String,
    pub links: Vec<String>,
    pub refreshed: bool,
}

fn detect_links(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|token| token.starts_with("http://") || token.starts_with("https://"))
        .map(|token| token.trim_end_matches(['.', ',', ';', ')', ']']))
        .map(|clean| clean.to_string())
        .collect()
}

// --- HELPER: Preprocess Image for CLIP ---
fn preprocess_for_clip(img: &image::DynamicImage) -> Array4<f32> {
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let mut input = Array4::zeros((1, 3, 224, 224));
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];

    for (x, y, pixel) in resized.pixels() {
        let r = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
        let g = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
        let b = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        input[[0, 0, y as usize, x as usize]] = r;
        input[[0, 1, y as usize, x as usize]] = g;
        input[[0, 2, y as usize, x as usize]] = b;
    }
    input
}

const TROCR_IMAGE_SIZE: u32 = 384;
const TROCR_BOS_TOKEN_ID: i64 = 0;
const TROCR_EOS_TOKEN_ID: i64 = 2;
const TROCR_PAD_TOKEN_ID: i64 = 1;
const TROCR_MAX_DECODE_STEPS: usize = 256;

fn preprocess_for_trocr(img: &image::DynamicImage) -> Array4<f32> {
    let resized = img
        .resize_exact(
            TROCR_IMAGE_SIZE,
            TROCR_IMAGE_SIZE,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();
    let mut input = Array4::zeros((1, 3, TROCR_IMAGE_SIZE as usize, TROCR_IMAGE_SIZE as usize));

    for (x, y, pixel) in resized.enumerate_pixels() {
        for (channel, value) in pixel.0.iter().enumerate() {
            let normalized = (*value as f32 / 255.0 - 0.5) / 0.5;
            input[[0, channel, y as usize, x as usize]] = normalized;
        }
    }

    input
}

fn tensor_value_to_array(value: &Value, label: &str) -> Result<ArrayD<f32>, String> {
    let (shape, data_view) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Failed to extract {label} tensor: {e}"))?;
    let owned = data_view.to_vec();
    ArrayD::from_shape_vec(shape.to_ixdyn(), owned)
        .map_err(|_| format!("{label} tensor shape was incompatible"))
}

fn extract_logits(outputs: &SessionOutputs<'_>) -> Result<ArrayD<f32>, String> {
    if let Some(value) = outputs.get("logits") {
        return tensor_value_to_array(value, "decoder logits");
    }

    if let Some((_, value_ref)) = outputs.iter().next() {
        return tensor_value_to_array(value_ref.deref(), "decoder logits");
    }

    Err("Decoder outputs missing logits".to_string())
}

fn select_next_token(logits: &ArrayD<f32>) -> Result<i64, String> {
    let logits3 = logits
        .clone()
        .into_dimensionality::<Ix3>()
        .map_err(|_| "Unexpected logits shape from decoder".to_string())?;
    let seq_len = logits3.shape()[1];
    if seq_len == 0 {
        return Err("Decoder returned empty sequence".to_string());
    }

    let last_slice = logits3.index_axis(Axis(1), seq_len - 1);
    let (token_idx, _) = last_slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .ok_or_else(|| "Decoder logits slice was empty".to_string())?;

    Ok(token_idx as i64)
}

fn refresh_past_cache_from_outputs(
    outputs: &SessionOutputs<'_>,
    cache: &mut HashMap<String, ArrayD<f32>>,
) -> Result<(), String> {
    cache.clear();
    for (name, value) in outputs.iter() {
        if !name.starts_with("present") {
            continue;
        }

        let tensor_array = tensor_value_to_array(value.deref(), name)
            .map_err(|e| format!("Failed to extract past tensor {name}: {e}"))?;
        let past_name = name.replacen("present", "past_key_values", 1);
        cache.insert(past_name, tensor_array);
    }

    if cache.is_empty() {
        return Err("Decoder outputs missing past key/value tensors".to_string());
    }

    Ok(())
}

fn build_decoder_init_inputs(
    session: &Session,
    token_sequence: &[i64],
    encoder_states: Array3<f32>,
) -> Result<HashMap<String, Value>, String> {
    if token_sequence.is_empty() {
        return Err("Decoder init requires at least one token".to_string());
    }

    let mut inputs = HashMap::new();

    let input_ids = Array2::from_shape_vec((1, token_sequence.len()), token_sequence.to_vec())
        .map_err(|_| "Failed to reshape decoder input ids".to_string())?;
    let input_tensor = Tensor::from_array(input_ids)
        .map_err(|e| format!("Failed to build decoder input tensor: {e}"))?;
    inputs.insert(session.inputs[0].name.clone(), input_tensor.into());

    let encoder_tensor = Tensor::from_array(encoder_states)
        .map_err(|e| format!("Failed to build encoder hidden tensor: {e}"))?;
    inputs.insert(session.inputs[1].name.clone(), encoder_tensor.into());

    Ok(inputs)
}

fn build_decoder_with_past_inputs(
    session: &Session,
    next_token: i64,
    cache: &HashMap<String, ArrayD<f32>>,
) -> Result<HashMap<String, Value>, String> {
    let mut inputs = HashMap::new();

    let tokens = Array2::from_shape_vec((1, 1), vec![next_token])
        .map_err(|_| "Failed to reshape incremental decoder ids".to_string())?;
    let ids_tensor = Tensor::from_array(tokens)
        .map_err(|e| format!("Failed to build incremental ids tensor: {e}"))?;
    inputs.insert(session.inputs[0].name.clone(), ids_tensor.into());

    for input in session.inputs.iter().skip(1) {
        let tensor_data = cache
            .get(&input.name)
            .ok_or_else(|| format!("Missing cached tensor for {}", input.name))?;
        let tensor = Tensor::from_array(tensor_data.clone())
            .map_err(|e| format!("Failed to rebuild past tensor {}: {e}", input.name))?;
        inputs.insert(input.name.clone(), tensor.into());
    }

    Ok(inputs)
}

fn decoder_step(
    outputs: &SessionOutputs<'_>,
    cache: &mut HashMap<String, ArrayD<f32>>,
) -> Result<i64, String> {
    let logits = extract_logits(outputs)?;
    refresh_past_cache_from_outputs(outputs, cache)?;
    select_next_token(&logits)
}

fn run_trocr_pipeline(
    ocr_state: &mut OcrState,
    image_path: &Path,
    timestamp: String,
) -> Result<OcrCacheEntry, String> {
    if !image_path.exists() {
        return Err(format!(
            "OCR source file not found at {}",
            image_path.display()
        ));
    }

    let image =
        image::open(image_path).map_err(|e| format!("Failed to open screenshot for OCR: {e}"))?;
    let pixel_values = preprocess_for_trocr(&image);

    let encoder_input = Tensor::from_array(pixel_values)
        .map_err(|e| format!("Failed to create encoder tensor: {e}"))?;
    let encoder_input_name = ocr_state.encoder_session.inputs[0].name.clone();
    let encoder_outputs = ocr_state
        .encoder_session
        .run(ort::inputs![encoder_input_name => encoder_input])
        .map_err(|e| format!("Encoder run failed: {e}"))?;

    let encoder_hidden_array = if let Some(value) = encoder_outputs.get("last_hidden_state") {
        tensor_value_to_array(value, "encoder hidden state")
    } else if let Some((_, value_ref)) = encoder_outputs.iter().next() {
        tensor_value_to_array(value_ref.deref(), "encoder hidden state")
    } else {
        Err("Encoder outputs missing hidden state".to_string())
    }?;
    let encoder_hidden = encoder_hidden_array
        .into_dimensionality::<Ix3>()
        .map_err(|_| "Unexpected encoder hidden shape".to_string())?;

    let mut cache: HashMap<String, ArrayD<f32>> = HashMap::new();
    let mut decoded_tokens: Vec<i64> = Vec::new();
    let mut current_token = TROCR_BOS_TOKEN_ID;
    let mut encoder_states_option = Some(encoder_hidden);

    for step in 0..TROCR_MAX_DECODE_STEPS {
        let outputs = if step == 0 {
            let encoder_states = encoder_states_option
                .take()
                .ok_or_else(|| "Missing encoder states".to_string())?;
            let inputs = build_decoder_init_inputs(
                &ocr_state.decoder_init_session,
                &[current_token],
                encoder_states,
            )?;
            ocr_state
                .decoder_init_session
                .run(inputs)
                .map_err(|e| format!("Decoder init run failed: {e}"))?
        } else {
            let inputs = build_decoder_with_past_inputs(
                &ocr_state.decoder_with_past_session,
                current_token,
                &cache,
            )?;
            ocr_state
                .decoder_with_past_session
                .run(inputs)
                .map_err(|e| format!("Decoder with past run failed: {e}"))?
        };

        let next_token = decoder_step(&outputs, &mut cache)?;

        if next_token == TROCR_EOS_TOKEN_ID || next_token == TROCR_PAD_TOKEN_ID {
            break;
        }

        decoded_tokens.push(next_token);
        current_token = next_token;
    }

    let filtered_tokens: Vec<i64> = decoded_tokens
        .into_iter()
        .filter(|token| *token != TROCR_BOS_TOKEN_ID && *token != TROCR_PAD_TOKEN_ID)
        .collect();

    let decoded_text = if filtered_tokens.is_empty() {
        String::new()
    } else {
        ocr_state
            .tokenizer
            .decode(&filtered_tokens)
            .map_err(|e| format!("SentencePiece decode failed: {e}"))?
    };

    let cleaned_text = decoded_text.replace('\n', " ").trim().to_string();
    let links = detect_links(&cleaned_text);

    Ok(OcrCacheEntry {
        timestamp,
        text: cleaned_text,
        links,
    })
}

// --- HELPER: Initialize LanceDB ---
async fn init_lancedb(app_handle: &tauri::AppHandle) -> Result<(PathBuf, Table), String> {
    let app_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let lancedb_path = app_dir.join("lancedb_store");

    if !lancedb_path.exists() {
        fs::create_dir_all(&lancedb_path).map_err(|e| e.to_string())?;
    }

    let db = lancedb::connect(lancedb_path.to_str().unwrap())
        .execute()
        .await
        .map_err(|e| e.to_string())?;

    const VECTOR_SIZE: i32 = 512;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VECTOR_SIZE,
            ),
            true,
        ),
    ]));

    let table = db
        .create_table("memories", RecordBatchIterator::new(vec![], schema.clone()))
        .execute()
        .await;

    let table = match table {
        Ok(t) => t,
        Err(_) => db
            .open_table("memories")
            .execute()
            .await
            .map_err(|e| e.to_string())?,
    };

    println!("DEBUG: LanceDB initialized at {:?}", lancedb_path);
    Ok((app_dir, table))
}

// --- HELPER: Save Memory to LanceDB ---
async fn add_memory(
    table: &Table,
    id: String,
    timestamp: String,
    path: String,
    embedding: Vec<f32>,
) {
    const VECTOR_SIZE: i32 = 512;

    let id_array = arrow::array::StringArray::from(vec![id]);
    let ts_array = arrow::array::StringArray::from(vec![timestamp]);
    let path_array = arrow::array::StringArray::from(vec![path]);

    let vector_values = arrow::array::Float32Array::from(embedding);
    let fixed_size_list = arrow::array::FixedSizeListArray::new(
        Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::Float32,
            true,
        )),
        VECTOR_SIZE,
        Arc::new(vector_values),
        None,
    );

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("timestamp", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("file_path", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new(
            "vector",
            arrow::datatypes::DataType::FixedSizeList(
                Arc::new(arrow::datatypes::Field::new(
                    "item",
                    arrow::datatypes::DataType::Float32,
                    true,
                )),
                VECTOR_SIZE,
            ),
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(ts_array),
            Arc::new(path_array),
            Arc::new(fixed_size_list),
        ],
    )
    .unwrap();

    let schema = table.schema().await.unwrap();
    let _ = table
        .add(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
        .execute()
        .await;
}

// --- MAIN LOOP ---
async fn start_screen_capture_loop(
    app_handle: tauri::AppHandle,
    app_dir: PathBuf,
    mut session: Session,
    table: Table,
) {
    println!("DEBUG: Starting Mnemosyne Memory Core...");
    let images_dir = app_dir.join("snapshots");
    if !images_dir.exists() {
        let _ = fs::create_dir_all(&images_dir);
    }

    loop {
        let screens = Screen::all().unwrap_or_default();

        if let Some(screen) = screens.first() {
            if let Ok(image_buffer) = screen.capture() {
                // UI Stream
                let mut buffer = Vec::new();
                let mut cursor = Cursor::new(&mut buffer);
                let dynamic_image = image::DynamicImage::ImageRgba8(image_buffer);

                let thumbnail = dynamic_image.thumbnail(800, 600);
                if thumbnail
                    .write_to(&mut cursor, ImageOutputFormat::Png)
                    .is_ok()
                {
                    let base64_string = base64::engine::general_purpose::STANDARD.encode(&buffer);
                    let data_url = format!("data:image/png;base64,{}", base64_string);
                    let _ = app_handle.emit("new-screenshot", data_url);
                }

                // AI Inference
                let input_tensor_values = preprocess_for_clip(&dynamic_image);
                let input_tensor = Tensor::from_array(input_tensor_values).unwrap();
                let input_name = session.inputs[0].name.clone();

                let mut embedding_vec: Vec<f32> = Vec::new();

                match session.run(ort::inputs![input_name => input_tensor]) {
                    Ok(outputs) => {
                        let (_, output_value) = outputs.iter().next().unwrap();
                        let output_tensor = output_value.try_extract_tensor::<f32>().unwrap();
                        let (_, embedding) = output_tensor;

                        embedding_vec = embedding.to_vec();
                    }
                    Err(e) => println!("Inferencing Error: {}", e),
                }

                // Persistence
                if !embedding_vec.is_empty() {
                    let id = uuid::Uuid::new_v4().to_string();
                    let now = Local::now();
                    let timestamp_str = now.format("%Y-%m-%d %H:%M:%S").to_string();
                    let filename = format!("{}.png", id);
                    let file_path = images_dir.join(&filename);

                    if dynamic_image.save(&file_path).is_ok() {
                        let path_string = file_path.to_string_lossy().to_string();

                        add_memory(
                            &table,
                            id,
                            timestamp_str.clone(),
                            path_string,
                            embedding_vec,
                        )
                        .await;
                        println!("DEBUG: Memory Encoded & Saved. [{}]", timestamp_str);
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

// --- SEARCH + OCR COMMANDS ---
fn text_to_tensor(text: &str) -> ndarray::Array2<i64> {
    let _ = text;
    let token_ids: Vec<i64> = vec![49406, 11, 230, 2432, 234, 12, 11, 34, 45, 60, 484];
    let padding_size = 77 - token_ids.len();
    let tokens: Vec<i64> = token_ids
        .into_iter()
        .chain(std::iter::repeat(0).take(padding_size))
        .collect();

    ndarray::Array2::from_shape_vec((1, 77), tokens.into_iter().take(77).collect()).unwrap()
}

#[derive(serde::Serialize, Clone)]
pub struct SearchResult {
    id: String,
    timestamp: String,
    file_path: String,
    score: f32,
}

#[tauri::command]
async fn ocr_memory(
    state: tauri::State<'_, Mutex<AppState>>,
    memory_id: String,
    file_path: String,
    timestamp: String,
) -> Result<OcrResult, String> {
    println!("DEBUG: Received OCR request for memory {}", memory_id);

    let mut app_state = state.lock().unwrap();
    let ocr_state = app_state
        .ocr_state
        .as_mut()
        .ok_or_else(|| "TrOCR resources not yet initialized".to_string())?;

    if let Some(entry) = ocr_state.cache.get(&memory_id) {
        if entry.timestamp == timestamp {
            return Ok(OcrResult {
                id: memory_id,
                timestamp: entry.timestamp.clone(),
                text: entry.text.clone(),
                links: entry.links.clone(),
                refreshed: false,
            });
        }
    }

    let image_path = Path::new(&file_path);
    let fresh_entry = run_trocr_pipeline(ocr_state, image_path, timestamp.clone())?;
    ocr_state
        .cache
        .insert(memory_id.clone(), fresh_entry.clone());

    Ok(OcrResult {
        id: memory_id,
        timestamp: fresh_entry.timestamp.clone(),
        text: fresh_entry.text.clone(),
        links: fresh_entry.links.clone(),
        refreshed: true,
    })
}

#[tauri::command]
async fn search_memories(
    state: tauri::State<'_, Mutex<AppState>>,
    query: String,
) -> Result<Vec<SearchResult>, String> {
    println!("DEBUG: Received search query: '{}'", query);

    let (table_clone, text_embedding) = {
        let mut app_state = state.lock().unwrap();
        let table_clone = app_state
            .db_table
            .as_ref()
            .ok_or("Database not yet initialized")?
            .clone();
        let text_session = app_state
            .text_session
            .as_mut()
            .ok_or("Text Model not yet loaded")?;

        let text_input_array = text_to_tensor(&query);
        let text_input_tensor = Tensor::from_array(text_input_array).unwrap();

        let input_name = text_session.inputs[0].name.clone();

        let text_outputs = text_session
            .run(ort::inputs![input_name => text_input_tensor])
            .map_err(|e| e.to_string())?;

        let (_, output_value) = text_outputs.iter().next().unwrap();
        let output_tensor = output_value.try_extract_tensor::<f32>().unwrap();
        let (_, embedding_slice) = output_tensor;
        let text_embedding: Vec<f32> = embedding_slice.to_vec();

        (table_clone, text_embedding)
    };

    let results = table_clone
        .query()
        .nearest_to(text_embedding)
        .map_err(|e| format!("Query builder error: {}", e))?
        .limit(10)
        .select(Select::columns(&["id", "timestamp", "file_path"]))
        .execute()
        .await
        .map_err(|e| e.to_string())?;

    let mut search_results: Vec<SearchResult> = Vec::new();
    let mut stream = results;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result.map_err(|e| e.to_string())?;

        let id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let ts_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let fp_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let id = id_col.value(i).to_string();
            let timestamp = ts_col.value(i).to_string();
            let file_path = fp_col.value(i).to_string();
            let score = 0.0;

            search_results.push(SearchResult {
                id,
                timestamp,
                file_path,
                score,
            });
        }
    }

    println!(
        "DEBUG: Search executed. Found {} results.",
        search_results.len()
    );
    Ok(search_results)
}

fn prepare_onnxruntime(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let runtime_dir = app
        .path()
        .resolve("resources/onnxruntime", BaseDirectory::Resource)?;
    let dll_path = runtime_dir.join("onnxruntime.dll");

    if !dll_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Pinned ONNX Runtime DLL missing at {}. Run `npm run sync-onnxruntime` to regenerate it.",
                dll_path.display()
            ),
        )
        .into());
    }

    let normalize = |path: &PathBuf| -> Result<String, io::Error> {
        let raw = path.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Runtime path contains invalid UTF-8")
        })?;
        Ok(raw.trim_start_matches("\\\\?\\").to_string())
    };

    let runtime_dir_str = normalize(&runtime_dir)?;
    let dll_path_str = normalize(&dll_path)?;

    std::env::set_var("ORT_DYLIB_PATH", &dll_path_str);

    let path_value = std::env::var("PATH").unwrap_or_default();
    let already_present = path_value
        .split(';')
        .any(|segment| segment.eq_ignore_ascii_case(&runtime_dir_str));
    if !already_present {
        let updated = if path_value.is_empty() {
            runtime_dir_str.clone()
        } else {
            format!("{};{}", runtime_dir_str, path_value)
        };
        std::env::set_var("PATH", updated);
    }

    ort::init().with_name("Mnema").commit().map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to initialize ONNX Runtime: {err}"),
        )
    })?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState {
            db_table: None,
            text_session: None,
            ocr_state: None,
        }))
        .setup(|app| {
            prepare_onnxruntime(&app.handle())?;

            let handle = app.handle().clone();

            let vision_path = app
                .path()
                .resolve("resources/clip-vision.onnx", BaseDirectory::Resource)
                .expect("failed to resolve vision model");
            let text_path = app
                .path()
                .resolve("resources/clip-text.onnx", BaseDirectory::Resource)
                .expect("failed to resolve text model");

            println!("DEBUG: Loading CLIP Vision Model...");
            let vision_session = Session::builder()?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(vision_path)?;

            println!("DEBUG: Loading CLIP Text Model...");
            let text_session = Session::builder()?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(text_path)?;

            let app_state_mutex = app.state::<Mutex<AppState>>();
            app_state_mutex.lock().unwrap().text_session = Some(text_session);

            match load_trocr_resources(&handle) {
                Ok(ocr_state) => {
                    app_state_mutex.lock().unwrap().ocr_state = Some(ocr_state);
                    println!("DEBUG: TrOCR resources loaded and ready.");
                }
                Err(err) => {
                    println!("WARN: Unable to load TrOCR resources: {err}");
                }
            }

            let handle_for_state = handle.clone();

            tauri::async_runtime::spawn(async move {
                match init_lancedb(&handle).await {
                    Ok((app_dir, table)) => {
                        handle_for_state
                            .state::<Mutex<AppState>>()
                            .lock()
                            .unwrap()
                            .db_table = Some(table.clone());

                        start_screen_capture_loop(handle, app_dir, vision_session, table).await;
                    }
                    Err(e) => println!("CRITICAL ERROR: LanceDB Init failed: {}", e),
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search_memories, ocr_memory])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn build_directml_session(path: &Path) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_execution_providers([DirectMLExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)
}

fn load_trocr_resources(app: &tauri::AppHandle) -> Result<OcrState, Box<dyn std::error::Error>> {
    let trocr_dir = app
        .path()
        .resolve("resources/trocr-base", BaseDirectory::Resource)?;
    let encoder_path = trocr_dir.join("encoder_model_quantized.onnx");
    let decoder_init_path = trocr_dir.join("decoder_model_quantized.onnx");
    let decoder_with_past_path = trocr_dir.join("decoder_with_past_model_quantized.onnx");
    let tokenizer_path = trocr_dir.join("sentencepiece.bpe.model");

    for required in [
        &encoder_path,
        &decoder_init_path,
        &decoder_with_past_path,
        &tokenizer_path,
    ] {
        if !required.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Missing TrOCR asset at {}", required.display()),
            )
            .into());
        }
    }

    let encoder_session = build_directml_session(&encoder_path)?;
    let decoder_init_session = build_directml_session(&decoder_init_path)?;
    let decoder_with_past_session = build_directml_session(&decoder_with_past_path)?;

    let tokenizer_path_str = tokenizer_path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Other,
            "Tokenizer path contains invalid UTF-8",
        )
    })?;
    let tokenizer = SentencePieceProcessor::load(tokenizer_path_str).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to load tokenizer from {}: {e}", tokenizer_path.display()),
        )
    })?;

    Ok(OcrState {
        encoder_session,
        decoder_init_session,
        decoder_with_past_session,
        tokenizer,
        cache: HashMap::new(),
    })
}
