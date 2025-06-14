use crate::utils::{count_value};
use crate::cell::CellType;
pub struct Status{
    pub cancer_percent: f32,
    pub cancer_cells: usize,
    pub normal_percent: f32,
    pub normal_cells: usize,
    pub wbc_percent: f32,
    pub wbc_cells: usize,
    pub dead_percent: f32,
    pub dead_cells: usize,
    pub regenerated_percent: f32,
    pub regenerated_cells: usize,
    pub ctt_percent: f32,
    pub ctt: usize,
}

pub fn status_calculate(data: &Vec<u32>, wbc_data: &Vec<f32>, ctt_data: &Vec<f32>) -> Status {
    let total_cells = data.len();
    let cancer_cells = count_value(data, CellType::CancerCell as u32);
    let normal_cells = count_value(data, CellType::NormalCell as u32);
    let regenerated_cells = count_value(data, CellType::RegeneratedCell as u32);
    let wbc_cells = wbc_data.iter().sum::<f32>() as usize;
    let dead_cells = count_value(data, CellType::DeadCell as u32);
    
    // 计算CTT浓度总和和有效CTT位置数量
    let ctt_concentration_sum = ctt_data.iter().sum::<f32>();
    let ctt = ctt_data.iter().filter(|&&x| x > 0.01).count(); // 只计算有效浓度的位置
    
    let cancer_percent = cancer_cells as f32 / total_cells as f32 * 100.0;
    let normal_percent = normal_cells as f32 / total_cells as f32 * 100.0;
    let wbc_percent = wbc_cells as f32 / total_cells as f32 * 100.0;
    let dead_percent = dead_cells as f32 / total_cells as f32 * 100.0;
    let regenerated_percent = regenerated_cells as f32 / total_cells as f32 * 100.0;
    let ctt_percent = ctt_concentration_sum / total_cells as f32 * 100.0; // 使用浓度总和计算百分比
    
    Status{
        cancer_percent,
        cancer_cells,
        normal_percent,
        normal_cells,
        wbc_percent,
        wbc_cells,
        dead_percent,
        dead_cells,
        regenerated_percent,
        regenerated_cells,
        ctt_percent,
        ctt
    }
}
