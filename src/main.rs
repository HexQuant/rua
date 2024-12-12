use chrono::{DateTime, Utc};
use reqwest::{Client, Error};
use serde::{Deserialize, Deserializer};
use std::io::Write;
use std::path::Path;
// use std::thread::sleep;
use std::time::Duration;
use tqdm::pbar;

#[derive(Debug, Deserialize)]
struct Area {
    #[serde(skip_deserializing)]
    time_index: DateTime<Utc>,
    hash: String,
    area: f64,
    #[serde(deserialize_with = "str_to_f64")]
    percent: f64,
    #[serde(rename = "type")]
    area_type: String,
}

#[derive(Debug, Deserialize)]
struct AreaItem {
    id: i64,
    // #[serde(rename = "description")]
    // description_ua: String,
    // #[serde(rename = "descriptionEn")]
    // description_en: String,
    #[serde(rename = "updatedAt")]
    updated_at: DateTime<Utc>,
    datetime: String,
    status: bool,
    #[serde(rename = "createdAt")]
    created_at: DateTime<Utc>,
}

fn str_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    s.parse::<f64>().map_err(serde::de::Error::custom)
}

async fn fetch_url(
    client: &Client,
    timestamp: i64,
    max_retries: u32,
    delay: Duration,
) -> Result<String, Error> {
    let url = format!("https://deepstatemap.live/api/history/{timestamp}/areas");
    let mut last_error: Option<Error> = None;
    for attempt in 0..max_retries {
        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return response.text().await;
                } else {
                    eprintln!("Attempt {} failed: HTTP {}", attempt + 1, response.status());
                }
            }
            Err(err) => {
                eprintln!("Attempt {} failed: {:?}", attempt + 1, err);
                last_error = Some(err);
            }
        }
        if attempt < max_retries - 1 {
            tokio::time::sleep(delay).await;
        }
    }
    Err(last_error.unwrap())
}

fn get_timestamps() -> Result<String, Error> {
    let client = reqwest::blocking::Client::new();
    let time_history_url = "https://deepstatemap.live/api/history/public";
    match client.get(time_history_url).send() {
        Ok(response) => {
            if response.status().is_success() {
                return response.text();
            } else {
                return Err(response.error_for_status().unwrap_err());
            }
        }
        Err(err) => {
            panic!("Failed to fetch the URL: {}", err);
        }
    }
}

fn to_csv(areas: Vec<Area>, file_path: &Path) {
    let mut file = std::fs::File::create(file_path).unwrap();
    let head_str = "time_index,hash,area,percent,area_type\n";
    file.write_all(head_str.as_bytes()).unwrap();
    for area in areas {
        let line = format!(
            "{},{},{},{},{}\n",
            area.time_index, area.hash, area.area, area.percent, area.area_type
        );
        file.write_all(line.as_bytes()).unwrap();
    }
}

#[tokio::main]
async fn main() {
    println!("RUA - Dynamic transition of territory in the Russian-Ukrainian conflict");
    let max_retries = 10;
    let delay = Duration::from_secs(2);

    // Загрузка временных меток
    println!("Fetching timestamps...");
    let json_data = get_timestamps().unwrap();
    let result: Vec<AreaItem> =
        serde_json::from_str(&json_data).expect("Failed to deserialize JSON");

    // Загрузка площадей
    let mut areas = Vec::with_capacity(5000);
    let client = Client::new();
    let mut pbar = pbar(Some(result.len()));
    for area_item in result {
        let timestamp = area_item.id;

        match fetch_url(&client, timestamp, max_retries, delay).await {
            Ok(content) => {
                let mut area: Vec<Area> =
                    serde_json::from_str(&content).expect("Failed to deserialize JSON");
                for a in area.iter_mut() {
                    a.time_index = DateTime::<Utc>::from_timestamp(timestamp, 0).unwrap();
                }
                areas.extend(area);
                pbar.update(1).unwrap();
            }
            Err(err) => eprintln!("Failed to fetch the URL: {:?}", err),
        }
    }
    to_csv(areas, Path::new("data/area_history.csv"));
}
