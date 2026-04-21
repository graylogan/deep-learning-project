// Written by Brendan Westley for MST CS5480 Group Project

use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    sync::mpsc,
    thread,
};

use fast_image_resize::{IntoImageView, ResizeOptions, Resizer, images::Image};
use image::{ImageEncoder, ImageReader, codecs::tiff::TiffEncoder};
use serde_json::Value;

// {features: [{properties: {
//     "bounds_imcoords": "2671,838,2714,869",
//     "edited_by": "wwoscarbecerril",
//     "cat_id": "1040010028371A00",
//     "type_id": 73,
//     "ingest_time": "2017/07/24 13:20:38.280+00",
//     "index_right": 2356,
//     "image_id": "2355.tif",
//     "point_geom": "0101000020E610000060DE147608A256C05D6A384C61222D40",
//     "feature_id": 377368,
//     "grid_file": "Grid2.shp"
// }}]}
#[derive(Clone)]
struct Feature {
    x1: i32,          // 2671
    y1: i32,          // 838
    x2: i32,          // 2714
    y2: i32,          // 869
    type_id: u64,     // 73
    image_id: String, // "2355.tif"
    feature_id: u64,  // 377368
}

fn get_feature_debug(feature: &Value) -> Option<Feature> {
    match get_feature(feature) {
        Ok(f) => Some(f),
        Err(e) => {
            println!("Error parsing feature: {e}\n{}", feature.to_string());
            None
        }
    }
}

fn get_feature(feature: &Value) -> Result<Feature, &'static str> {
    let properties = feature
        .as_object()
        .ok_or("feature not object")?
        .get("properties")
        .ok_or("feature missing properties")?
        .as_object()
        .ok_or("properties not object")?;
    let mut bounds_imcoords = properties
        .get("bounds_imcoords")
        .ok_or("properties missing bounds_imcoords")?
        .as_str()
        .ok_or("bounds_imcoords not str")?
        .split_terminator(',');
    Ok(Feature {
        x1: bounds_imcoords
            .next()
            .ok_or("bounds_imcoords missing part 0")?
            .parse()
            .or(Err("bounds_imcoords part 0 not i64"))?,
        y1: bounds_imcoords
            .next()
            .ok_or("bounds_imcoords missing part 1")?
            .parse()
            .or(Err("bounds_imcoords part 1 not i64"))?,
        x2: bounds_imcoords
            .next()
            .ok_or("bounds_imcoords missing part 2")?
            .parse()
            .or(Err("bounds_imcoords part 2 not i64"))?,
        y2: bounds_imcoords
            .next()
            .ok_or("bounds_imcoords missing part 3")?
            .parse()
            .or(Err("bounds_imcoords part 3 not i64"))?,
        type_id: properties
            .get("type_id")
            .ok_or("properties missing type_id")?
            .as_u64()
            .ok_or("type_id not u64")?,
        image_id: properties
            .get("image_id")
            .ok_or("properties missing image_id")?
            .as_str()
            .ok_or("type_id not str")?
            .to_string(),
        feature_id: properties
            .get("feature_id")
            .ok_or("properties missing feature_id")?
            .as_u64()
            .ok_or("feature_id not u64")?,
    })
}

fn load_data() -> Option<Vec<Feature>> {
    let geojson: Value = serde_json::from_reader(BufReader::new(
        File::open("../../xView_train.geojson").ok()?,
    ))
    .ok()?;
    let geojson_features: &Vec<Value> = geojson.as_object()?.get("features")?.as_array()?;
    let features: Vec<Feature> = geojson_features
        .iter()
        .filter_map(get_feature_debug)
        .collect();
    Some(features)
}

fn crop_box(feature: &Feature, width: u32, height: u32) -> Option<(f64, f64, u32, u32)> {
    let width: i32 = width.try_into().unwrap_or(i32::MAX);
    let height: i32 = height.try_into().unwrap_or(i32::MAX);
    let x1 = feature.x1.clamp(0, width);
    let x2 = feature.x2.clamp(0, width);
    let y1 = feature.y1.clamp(0, height);
    let y2 = feature.y2.clamp(0, height);
    if x1 >= x2 || y1 >= y2 {
        return None;
    }
    Some((
        x1.into(),
        y1.into(),
        (x2 - x1).try_into().ok()?,
        (y2 - y1).try_into().ok()?,
    ))
}

fn chip_image(
    src_image: &image::DynamicImage,
    feature: &Feature,
    width: u32,
    height: u32,
) -> Option<()> {
    let crop_box = crop_box(&feature, src_image.width(), src_image.height())?;

    let mut dst_image = Image::new(width, height, src_image.pixel_type()?);
    let mut resizer = Resizer::new();
    resizer
        .resize(
            src_image,
            &mut dst_image,
            &ResizeOptions::new().crop(
                crop_box.0,
                crop_box.1,
                crop_box.2.into(),
                crop_box.3.into(),
            ),
        )
        .ok()?;
    let mut result_buf = BufWriter::new(
        File::create(format!(
            "../../output/{}/{}.tif",
            feature.type_id, feature.feature_id
        ))
        .ok()?,
    );
    TiffEncoder::new(&mut result_buf)
        .write_image(dst_image.buffer(), width, height, src_image.color().into())
        .ok()?;

    return Some(());
}

fn main() {
    println!("Loading features.");
    let mut features = load_data().unwrap();

    println!("Selecting features.");
    let mut type_ids = HashSet::new();
    for feature in features.iter() {
        type_ids.insert(feature.type_id);
    }
    for type_id in type_ids {
        let _ = std::fs::create_dir_all(format!("../../output/{type_id}"));
    }

    println!("Generating image chips.");
    let nproc: u64 = 10;
    let (tx, rx) = mpsc::channel();
    for i in 0..nproc {
        let tx_ = tx.clone();
        let tid = i;
        let features_ = features.clone();
        thread::spawn(move || {
            for (j, feature) in features_
                .iter()
                .enumerate()
                .skip(tid as usize)
                .step_by(nproc as usize)
            {
                let src_image = ImageReader::open(format!("../../images/{}", feature.image_id))
                    .ok()
                    .and_then(|r| r.decode().ok());
                if let Some(i) = src_image {
                    chip_image(&i, feature, 32, 32);
                }
                if j % 100 == 0 {
                    let _ = tx_.send(100);
                }
            }
        });
    }
    drop(tx);

    let mut lock = io::stdout().lock();
    let total: u64 = features.len() as u64;
    let mut processed: u64 = 0;
    for r in rx {
        processed += r;
        if processed % (nproc * 100) == 0 {
            let _ = write!(lock, "\r{processed}/{total} ({}%)", processed * 100 / total);
            let _ = lock.flush();
        }
    }
    println!();

    return;
}
