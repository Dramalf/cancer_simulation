use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Debug)]
pub struct Config {
    pub grid_width: u32,
    pub num_frames: u32,
    pub init_cancer_rate: f32,
    pub init_cancer_grid_width: u32,
    pub init_cancer_grid_num: u32,
    pub cancer_transformation_prob: f32,
    pub cell_regeneration_prob: f32,
    pub wbc_degeneration_prob: f32,
    pub wbc_regeneration_prob: f32,
    pub sleep_time: u32,
    pub ctt_effect: u32,
    pub regen_invincible_time: u32,
    pub cell_lifetime: u32
}

pub fn load_config() -> Config {
    let config_data = fs::read_to_string("config.json").expect("Unable to read config file");
    serde_json::from_str(&config_data).expect("Unable to parse config file")
}
