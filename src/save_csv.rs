use crate::status::Status;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

// 存储每个时间戳的所有比例数据
#[derive(Default)]
pub struct SimulationHistory {
    // 时间戳
    timestamps: Vec<u32>,
    // 癌细胞比例
    cancer_percents: Vec<f32>,
    // 白细胞浓度
    wbc_percents: Vec<f32>,
    // 死亡细胞比例
    dead_percents: Vec<f32>,
    // 正常细胞比例
    normal_percents: Vec<f32>,
    // 再生细胞比例
    regen_percents: Vec<f32>,
    // CTT浓度比例
    ctt_percents: Vec<f32>,
}

impl SimulationHistory {
    pub fn new() -> Self {
        Default::default()
    }

    // 添加一个新的数据点
    pub fn add_data_point(&mut self, timestamp: u32, status: &Status) {
        self.timestamps.push(timestamp);
        self.cancer_percents.push(status.cancer_percent);
        self.wbc_percents.push(status.wbc_percent);
        self.dead_percents.push(status.dead_percent);
        self.normal_percents.push(status.normal_percent);
        self.regen_percents.push(status.regenerated_percent);
        self.ctt_percents.push(status.ctt_percent);
    }

    // 导出为CSV文件
    pub fn export_to_csv(&self, filename: &str) -> io::Result<()> {
        // 确保输出目录存在
        let path = Path::new(filename);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut file = File::create(filename)?;
        
        // 写入CSV头
        writeln!(file, "timestamp,cancer_percent,wbc_percent,dead_percent,normal_percent,regen_percent,ctt_percent")?;
        
        // 写入数据
        for i in 0..self.timestamps.len() {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                self.timestamps[i],
                self.cancer_percents[i],
                self.wbc_percents[i],
                self.dead_percents[i],
                self.normal_percents[i],
                self.regen_percents[i],
                self.ctt_percents[i]
            )?;
        }
        
        Ok(())
    }
}
