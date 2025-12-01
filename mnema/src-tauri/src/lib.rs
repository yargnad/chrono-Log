use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tauri::{Emitter, Manager};
use screenshots::Screen;
use std::io::Cursor;
use image::{ImageOutputFormat, GenericImageView};
use base64::Engine;
use chrono::Local;

// --- AI & Math Imports ---
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel; 
use ort::value::Tensor;
use ort::execution_providers::DirectMLExecutionProvider;
use ndarray::{Array2, Array4};
use std::sync::Mutex; 

// --- DATABASE IMPORTS ---
use lancedb::Table;
use lancedb::query::{QueryBase, ExecutableQuery, Select}; // Import traits for query methods
use arrow::array::{RecordBatch, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use futures::StreamExt; // For async stream iteration

// --- GLOBAL STATE STRUCT (The Brain & Memory Manager) ---
pub struct AppState {
    pub db_table: Option<Table>,
    pub text_session: Option<Session>,
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

// --- HELPER: Initialize LanceDB ---
async fn init_lancedb(app_handle: &tauri::AppHandle) -> Result<(PathBuf, Table), String> {
    let app_dir = app_handle.path().app_data_dir().map_err(|e| e.to_string())?;
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
        Err(_) => db.open_table("memories").execute().await.map_err(|e| e.to_string())?,
    };

    println!("DEBUG: LanceDB initialized at {:?}", lancedb_path);
    Ok((app_dir, table))
}

// --- HELPER: Save Memory to LanceDB ---
async fn add_memory(table: &Table, id: String, timestamp: String, path: String, embedding: Vec<f32>) {
    const VECTOR_SIZE: i32 = 512;
    
    let id_array = arrow::array::StringArray::from(vec![id]);
    let ts_array = arrow::array::StringArray::from(vec![timestamp]);
    let path_array = arrow::array::StringArray::from(vec![path]);
    
    let vector_values = arrow::array::Float32Array::from(embedding);
    let fixed_size_list = arrow::array::FixedSizeListArray::new(
        Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::Float32, true)),
        VECTOR_SIZE,
        Arc::new(vector_values),
        None,
    );

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("timestamp", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("file_path", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("vector", arrow::datatypes::DataType::FixedSizeList(Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::Float32, true)), VECTOR_SIZE), true),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(ts_array),
            Arc::new(path_array),
            Arc::new(fixed_size_list),
        ],
    ).unwrap();

    let schema = table.schema().await.unwrap();
    let _ = table.add(RecordBatchIterator::new(vec![Ok(batch)], schema.clone())).execute().await;
}

// --- MAIN LOOP ---
async fn start_screen_capture_loop(app_handle: tauri::AppHandle, app_dir: PathBuf, mut session: Session, table: Table) {
    println!("DEBUG: Starting Mnemosyne Memory Core...");
    let images_dir = app_dir.join("snapshots");
    if !images_dir.exists() { let _ = fs::create_dir_all(&images_dir); }

    loop {
        let screens = Screen::all().unwrap_or_default();

        if let Some(screen) = screens.first() {
            if let Ok(image_buffer) = screen.capture() {
                // UI Stream
                let mut buffer = Vec::new();
                let mut cursor = Cursor::new(&mut buffer);
                let dynamic_image = image::DynamicImage::ImageRgba8(image_buffer);
                
                let thumbnail = dynamic_image.thumbnail(800, 600);
                if thumbnail.write_to(&mut cursor, ImageOutputFormat::Png).is_ok() {
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
                        
                        // println!("DEBUG: Memory Vector Generated (Size: {})", embedding_vec.len());
                    },
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
                        
                        // SAVE TO VECTOR DB
                        add_memory(&table, id, timestamp_str.clone(), path_string, embedding_vec).await;
                        println!("DEBUG: Memory Encoded & Saved. [{}]", timestamp_str);
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}


// --- THE SEARCH COMMAND (Frontend API) ---
// 1. Text preprocessing helper (Mock Tokenizer/Vector Converter)
fn text_to_tensor(text: &str) -> ndarray::Array2<i64> {
    let _ = text; 
    let token_ids: Vec<i64> = vec![
        49406, 
        11, 230, 2432, 234, 12, 11, 34, 45, 60, 484, 
    ];
    let padding_size = 77 - token_ids.len();
    let tokens: Vec<i64> = token_ids.into_iter().chain(std::iter::repeat(0).take(padding_size)).collect();
    
    ndarray::Array2::from_shape_vec((1, 77), tokens.into_iter().take(77).collect()).unwrap()
}

#[derive(serde::Serialize, Clone)]
pub struct SearchResult {
    id: String,
    timestamp: String,
    file_path: String,
    score: f32, // Lower score = more relevant
}

#[tauri::command]
async fn search_memories(state: tauri::State<'_, Mutex<AppState>>, query: String) -> Result<Vec<SearchResult>, String> {
    println!("DEBUG: Received search query: '{}'", query);

    // Extract what we need from state while holding the lock, then drop it before async operations
    let (table_clone, text_embedding) = {
        let mut app_state = state.lock().unwrap();
        let table_clone = app_state.db_table.as_ref().ok_or("Database not yet initialized")?.clone();
        let text_session = app_state.text_session.as_mut().ok_or("Text Model not yet loaded")?;

        // 1. Convert the user's text query into a search vector (Fingerprint)
        let text_input_array = text_to_tensor(&query);
        let text_input_tensor = Tensor::from_array(text_input_array).unwrap();
        
        let input_name = text_session.inputs[0].name.clone(); 

        let text_outputs = text_session.run(ort::inputs![input_name => text_input_tensor]).map_err(|e| e.to_string())?;
        
        let (_, output_value) = text_outputs.iter().next().unwrap();
        let output_tensor = output_value.try_extract_tensor::<f32>().unwrap();
        let (_, embedding_slice) = output_tensor;
        let text_embedding: Vec<f32> = embedding_slice.to_vec();

        (table_clone, text_embedding)
    }; // Lock is dropped here

    // 2. Query LanceDB for similar vectors (async operations happen after lock is dropped)
    let results = table_clone 
        .query() 
        .nearest_to(text_embedding) 
        .map_err(|e| format!("Query builder error: {}", e))? 
        .limit(10)
        .select(Select::columns(&["id", "timestamp", "file_path"])) 
        .execute()
        .await
        .map_err(|e| e.to_string())?;

    // 3. Format results for the Frontend
    let mut search_results: Vec<SearchResult> = Vec::new();
    let mut stream = results;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result.map_err(|e| e.to_string())?;
        
        let id_col = batch.column(0).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let ts_col = batch.column(1).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        let fp_col = batch.column(2).as_any().downcast_ref::<arrow::array::StringArray>().unwrap();

        for i in 0..batch.num_rows() {
            let id = id_col.value(i).to_string();
            let timestamp = ts_col.value(i).to_string();
            let file_path = fp_col.value(i).to_string();
            let score = 0.0; // Score not available in this query result format

            search_results.push(SearchResult { id, timestamp, file_path, score });
        }
    }

    println!("DEBUG: Search executed. Found {} results.", search_results.len());
    Ok(search_results)
}

// --- RUN ENTRY POINT ---
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    ort::init().with_name("Mnema").commit().expect("Failed to init ONNX");

    tauri::Builder::default()
        .manage(Mutex::new(AppState { 
            db_table: None, 
            text_session: None,
        }))
        .setup(|app| {
            let handle = app.handle().clone();

            let vision_path = app.path().resolve("resources/clip-vision.onnx", tauri::path::BaseDirectory::Resource)
                .expect("failed to resolve vision model");
            let text_path = app.path().resolve("resources/clip-text.onnx", tauri::path::BaseDirectory::Resource)
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

            // Clone handle again for use in async block
            let handle_for_state = handle.clone();

            tauri::async_runtime::spawn(async move {
                match init_lancedb(&handle).await {
                    Ok((app_dir, table)) => {
                        // Use handle to access state instead of app
                        handle_for_state.state::<Mutex<AppState>>().lock().unwrap().db_table = Some(table.clone());

                        start_screen_capture_loop(handle, app_dir, vision_session, table).await;
                    },
                    Err(e) => println!("CRITICAL ERROR: LanceDB Init failed: {}", e),
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search_memories])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}