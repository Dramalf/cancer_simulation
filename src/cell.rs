use rand::Rng;
use crate::utils::count_value;
pub enum CellType {
    NormalCell = 0,
    CancerCell = 1,
    DeadCell = 2,
    RegeneratedCell = 3,

    WhiteBloodCell = 4,
    /*
    0+4:白细胞移动到NormalCell
    1+4:白细胞移动到CancerCell
    2+4:白细胞移动到DeadCell
    3+4:白细胞移动到RegeneratedCell
     */
    TargetedTherapy = 8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellParams {
    pub cancer_transformation_prob: f32,
    pub cell_regeneration_prob: f32,
    pub wbc_degeneration_prob: f32,
    pub wbc_regeneration_prob: f32,
    pub time_stamp: u32,
    pub regen_invincible_time: u32,
    pub ctt_effect: u32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridUniforms {
    pub output_resolution: [u32; 2], // [width, height]
    pub grid_resolution: [u32; 2],   // [width, height]
}

pub fn init_wbc(cell_grid: &mut Vec<f32>, grid_width: u32, increment_rate: f32) {
    let mut rng = rand::rng();
    let total_wbc_cells = increment_rate * (grid_width * grid_width) as f32;
    let mut wbc_cells_placed = 0.0;

    while wbc_cells_placed < total_wbc_cells {
        let x = rng.random_range(0..grid_width);
        let y = rng.random_range(0..grid_width);
        let index = (y * grid_width + x) as usize;
        cell_grid[index] += 1.0;
        wbc_cells_placed += 1.0;
        // if cell_grid[index] == 0 || cell_grid[index] == 3 {
        //     cell_grid[index] = 4;
        //     wbc_cells_placed += 1;
        // }
    }
}

pub fn init_cell_grid(
    cell_grid: &mut Vec<u32>,
    grid_width: u32,
    init_cancer_rate: f32,
    init_cancer_grid_width: u32,
    init_cancer_grid_num: u32,
) {
    let mut rng = rand::rng();

    // Calculate total number of cancer cells needed
    let total_cancer_cells = (init_cancer_rate * (grid_width * grid_width) as f32) as u32;
    let mut cancer_cells_placed = 0;

    // Generate random center points for cancer clusters
    let mut center_points = Vec::new();
    for _ in 0..init_cancer_grid_num {
        let x =
            rng.random_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        let y =
            rng.random_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        center_points.push((x, y));
    }

    // Place cancer cells until we reach the target number
    while cancer_cells_placed < total_cancer_cells {
        // Randomly select one of the center points
        let (center_x, center_y) = center_points[rng.random_range(0..center_points.len())];

        // Generate random position within the square around the center point
        let x = center_x as i32
            + rng.random_range(
                -(init_cancer_grid_width as i32) / 2..=init_cancer_grid_width as i32 / 2,
            );
        let y = center_y as i32
            + rng.random_range(
                -(init_cancer_grid_width as i32) / 2..=init_cancer_grid_width as i32 / 2,
            );

        // Check if position is within grid bounds
        if x >= 0 && x < grid_width as i32 && y >= 0 && y < grid_width as i32 {
            let index = (y * grid_width as i32 + x) as usize;
            // Check if this position is not already a cancer cell
            if cell_grid[index] == 0 {
                cell_grid[index] = 1; // Mark as cancer cell
                cancer_cells_placed += 1;
            }
        }
    }
}

pub fn init_ctt(cell_grid: &mut Vec<u32>, grid_width: u32, ctt_positions: &Vec<(u32, u32)>,ctt_effect: u32) {
    for (x, y) in ctt_positions.iter() {
        cell_grid[(x + y * grid_width) as usize] = CellType::TargetedTherapy as u32+ctt_effect;
    }
}

