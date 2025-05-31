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
    #[serde(default = "default_init_strategy")]
    pub init_strategy: String,
}

// 为 init_strategy 提供默认值
fn default_init_strategy() -> String {
    "random".to_string()
}

pub fn load_config() -> Config {
    let config_data = fs::read_to_string("config.json").expect("Unable to read config file");
    let mut config: Config = serde_json::from_str(&config_data).expect("Unable to parse config file");
    
    // 验证 init_strategy 字段
    let valid_strategies = ["progressive", "random", "grid"];
    if !valid_strategies.contains(&config.init_strategy.as_str()) {
        eprintln!("Warning: Invalid init_strategy '{}'. Using 'random' as default.", config.init_strategy);
        config.init_strategy = "random".to_string();
    }
    
    config
}
