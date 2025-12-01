use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tauri::{Emitter, Manager};
use screenshots::Screen;
use std::io::Cursor;
use image::{ImageOutputFormat, GenericImageView};
use base64::Engine;
use chrono::Local;
use rusqlite::{params, Connection};

// ORT Imports
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel; 
use ort::value::Tensor;
use ort::execution_providers::DirectMLExecutionProvider;
use ndarray::Array4;

// --- HELPER: Turn an Image into a Tensor (CLIP Vision) ---
// CLIP typically uses: Resize to 224x224, then Normalize.
fn preprocess_for_clip(img: &image::DynamicImage) -> Array4<f32> {
    // 1. Resize to 224x224
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    
    // 2. Prepare the tensor structure (Batch Size 1, 3 Channels, 224 Height, 224 Width)
    let mut input = Array4::zeros((1, 3, 224, 224));
    
    // 3. CLIP specific Normalization Constants
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];

    // 4. Iterate pixels and fill the tensor
    for (x, y, pixel) in resized.pixels() {
        let r = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
        let g = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
        let b = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];

        // Standard ONNX format is NCHW (Batch, Channel, Height, Width)
        input[[0, 0, y as usize, x as usize]] = r;
        input[[0, 1, y as usize, x as usize]] = g;
        input[[0, 2, y as usize, x as usize]] = b;
    }

    input
}

// --- HELPER: Setup the Database ---
fn init_db(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    // 1. Resolve the path to the user's AppData folder
    let app_dir = app_handle.path().app_data_dir().map_err(|e| e.to_string())?;
    
    // 2. Create the directory if it doesn't exist
    if !app_dir.exists() {
        fs::create_dir_all(&app_dir).map_err(|e| e.to_string())?;
    }

    // 3. Connect to (or create) the SQLite database file
    let db_path = app_dir.join("chronolog.db");
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // 4. Create the table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL
        )",
        [],
    ).map_err(|e| e.to_string())?;

    Ok(app_dir)
}

// --- MAIN LOOP ---
async fn start_screen_capture_loop(app_handle: tauri::AppHandle, app_dir: PathBuf, mut session: Session) {
    println!("DEBUG: Starting Memory (Embedding) loop...");
    let images_dir = app_dir.join("snapshots");
    if !images_dir.exists() { let _ = fs::create_dir_all(&images_dir); }

    loop {
        let screens = Screen::all().unwrap_or_default();

        if let Some(screen) = screens.first() {
            if let Ok(image_buffer) = screen.capture() {
                // Prepare buffers
                let mut buffer = Vec::new();
                let mut cursor = Cursor::new(&mut buffer);
                let dynamic_image = image::DynamicImage::ImageRgba8(image_buffer);
                
                // --- STEP 1: UI Stream (Fast) ---
                let thumbnail = dynamic_image.thumbnail(800, 600);
                if thumbnail.write_to(&mut cursor, ImageOutputFormat::Png).is_ok() {
                     let base64_string = base64::engine::general_purpose::STANDARD.encode(&buffer);
                     let data_url = format!("data:image/png;base64,{}", base64_string);
                     let _ = app_handle.emit("new-screenshot", data_url);
                }

                // --- STEP 2: The "Thought" (CLIP Inference) ---
                let input_tensor_values = preprocess_for_clip(&dynamic_image);
                let input_tensor = Tensor::from_array(input_tensor_values).unwrap();
                
                // We dynamically grab the first input name from the model
                // This makes it work regardless of whether the model calls it "pixel_values" or "data"
                let input_name = session.inputs[0].name.clone();
                
                match session.run(ort::inputs![input_name => input_tensor]) {
                    Ok(outputs) => {
                        // Grab the embedding vector (the first output)
                        let (_, output_value) = outputs.iter().next().unwrap();
                        let output_tensor = output_value.try_extract_tensor::<f32>().unwrap();
                        let (_, embedding) = output_tensor; 
                        
                        // "embedding" is now a slice of 512 numbers representing the MEANING of your screen.
                        println!("DEBUG: Generated Memory Fingerprint (Vector Size: {})", embedding.len());
                        // Print the first few numbers just to prove it's mathing
                        if embedding.len() > 3 {
                            println!("DEBUG: Vector Start: [{:.4}, {:.4}, {:.4}...]", embedding[0], embedding[1], embedding[2]);
                        }
                    },
                    Err(e) => println!("Inferencing Error: {}", e),
                }

                // --- STEP 3: Persistence (Slow) ---
                let id = uuid::Uuid::new_v4().to_string();
                let now = Local::now();
                let timestamp_str = now.format("%Y-%m-%d %H:%M:%S").to_string();
                let filename = format!("{}.png", id);
                let file_path = images_dir.join(&filename);

                if dynamic_image.save(&file_path).is_ok() {
                    let db_path = app_dir.join("chronolog.db");
                    if let Ok(conn) = Connection::open(db_path) {
                        let path_string = file_path.to_string_lossy().to_string();
                        let _ = conn.execute(
                            "INSERT INTO snapshots (id, timestamp, file_path) VALUES (?1, ?2, ?3)",
                            params![id, timestamp_str, path_string],
                        );
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // 1. Initialize ONNX Runtime
    ort::init().with_name("ChronoLog").commit().expect("Failed to init ONNX");

    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();

            // 2. Load the CLIP Model
            // Ensure the file 'clip-vision.onnx' is in src-tauri/resources/
            let resource_path = app.path().resolve("resources/clip-vision.onnx", tauri::path::BaseDirectory::Resource)
                .expect("failed to resolve resource");

            println!("DEBUG: Loading CLIP Model...");
            let session = Session::builder()? 
                // Attempt Hardware Acceleration
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(resource_path)?;

            println!("DEBUG: CLIP Brain loaded.");
            
            match init_db(&handle) {
                Ok(app_dir) => {
                     tauri::async_runtime::spawn(async move {
                        start_screen_capture_loop(handle, app_dir, session).await;
                    });
                }
                Err(e) => println!("CRITICAL ERROR: DB Init failed: {}", e),
            }
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}