use rand::Rng;
use rand::prelude::*;

use crate::utils::count_value;
use pmj::{generate, Sample};
use rand::rngs::SmallRng;

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
    let mut rng = rand::thread_rng();
    let total_wbc_cells = increment_rate * (grid_width * grid_width) as f32;
    let mut wbc_cells_placed = 0.0;

    while wbc_cells_placed < total_wbc_cells {
        let x = rng.gen_range(0..grid_width);
        let y = rng.gen_range(0..grid_width);
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
    let mut rng = rand::thread_rng();

    // Calculate total number of cancer cells needed
    let total_cancer_cells = (init_cancer_rate * (grid_width * grid_width) as f32) as u32;
    let mut cancer_cells_placed = 0;

    // Generate random center points for cancer clusters
    let mut center_points = Vec::new();
    for _ in 0..init_cancer_grid_num {
        let x =
            rng.gen_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        let y =
            rng.gen_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        center_points.push((x, y));
    }

    // Place cancer cells until we reach the target number
    while cancer_cells_placed < total_cancer_cells {
        // Randomly select one of the center points
        let (center_x, center_y) = center_points[rng.gen_range(0..center_points.len())];

        // Generate random position within the square around the center point
        let x = center_x as i32
            + rng.gen_range(
                -(init_cancer_grid_width as i32) / 2..=init_cancer_grid_width as i32 / 2,
            );
        let y = center_y as i32
            + rng.gen_range(
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
pub fn progressive_multi_jittered_sampling(
    cell_grid: &mut Vec<u32>,
    grid_width: u32,
    init_cancer_rate: f32,
    // PMJ-specific parameters you might want to expose or tune:

) {
    let blue_noise_retry_count = 4;
    let seed = 0;
    let grid_size = grid_width * grid_width;
    if grid_size == 0 {
        cell_grid.clear();
        return;
    }
    let total_cancer_cells = (init_cancer_rate * grid_size as f32).round() as u32;

    if total_cancer_cells > grid_size {
        panic!("Total cancer cells ({}) cannot exceed the grid size ({}).", total_cancer_cells, grid_size);
    }

    // Ensure the cell_grid has the correct size and is initialized to 0 (healthy)
    if cell_grid.len() != grid_size as usize {
        *cell_grid = vec![0; grid_size as usize];
    } else {
        for i in 0..grid_size as usize {
            cell_grid[i] = 0; // Reset to healthy
        }
    }

    if total_cancer_cells == 0 {
        return; // No cancer cells to initialize
    }

    // Initialize a random number generator
    let mut rng = SmallRng::seed_from_u64(seed);

    // Generate PMJ sample points using the pmj crate's logic [cite: 1, 5]
    // The generate function takes sample_count, blue_noise_retry_count, and the rng[cite: 5].
    let pmj_samples = generate(
        total_cancer_cells as usize,
        blue_noise_retry_count,
        &mut rng,
    );

    let mut actual_cancer_cells_placed = 0;
    // Map these continuous PMJ samples to discrete grid cells
    for pmj_sample in pmj_samples {
        let x_continuous = pmj_sample.x(); // Returns f32 in [0, 1) [cite: 5]
        let y_continuous = pmj_sample.y(); // Returns f32 in [0, 1) [cite: 5]

        // Convert continuous [0, 1) coordinates to discrete grid coordinates [0, grid_width - 1]
        // Using .floor() ensures that values map correctly to indices.
        let mut grid_x = (x_continuous * grid_width as f32).floor() as u32;
        let mut grid_y = (y_continuous * grid_width as f32).floor() as u32;

        // Clamp coordinates to be within bounds, just in case of floating point quirks
        // or if pmj_sample.x() or .y() could theoretically yield exactly 1.0 (docs say [0,1))
        if grid_x >= grid_width {
            grid_x = grid_width - 1;
        }
        if grid_y >= grid_width {
            grid_y = grid_width - 1;
        }

        let index = (grid_y * grid_width + grid_x) as usize;

        // Mark the cell as cancerous (e.g., value 1)
        // Handle potential collisions: if multiple PMJ samples map to the same discrete cell.
        // For this implementation, if a cell is already cancerous, we don't increment the count.
        // This might lead to slightly fewer than `total_cancer_cells` if collisions are high.
        if cell_grid[index] == 0 {
            cell_grid[index] = 1; // Mark as cancer cell
            actual_cancer_cells_placed += 1;
        }
    }
    
    // Optional: If you strictly need *exactly* total_cancer_cells and collisions occurred,
    // you might need a more complex strategy here, e.g., randomly placing the remaining
    // few cells in unoccupied spots. However, this would deviate from pure PMJS placement.
    // For now, we accept the number placed via direct PMJS mapping.
    if actual_cancer_cells_placed < total_cancer_cells {
        // This can happen due to discretization collisions if total_cancer_cells is high
        // relative to grid_size, or if grid_width is small.
        // println!(
        //     "Warning: Placed {} cancer cells due to discretization, target was {}.",
        //     actual_cancer_cells_placed, total_cancer_cells
        // );
        // One simple strategy to fill the remainder randomly (if strictly needed):
        let mut remaining_to_place = total_cancer_cells - actual_cancer_cells_placed;
        if remaining_to_place > 0 {
            let mut empty_indices: Vec<usize> = cell_grid.iter().enumerate()
                .filter(|&(_, &val)| val == 0)
                .map(|(i, _)| i)
                .collect();
            
            use rand::seq::SliceRandom; // Requires rand crate
            empty_indices.shuffle(&mut rng);

            for i in 0..remaining_to_place.min(empty_indices.len() as u32) {
                cell_grid[empty_indices[i as usize]] = 1;
            }
        }
    }
}
// pub fn progressive_multi_jittered_sampling(
//     cell_grid: &mut Vec<u32>,
//     grid_width: u32,
//     init_cancer_rate: f32,
// ) {
//     let mut rng = rand::rng();
//     let total_cancer_cells = (init_cancer_rate * (grid_width * grid_width) as f32) as u32;
    
//     // 确保至少有一个癌细胞要放置
//     if total_cancer_cells == 0 {
//         return; // 如果不需要放置癌细胞，直接返回
//     }
    
//     let mut cancer_cells_placed = 0;

//     // 计算有多少层划分 (n×n 划分)
//     let n_samples = total_cancer_cells.min(grid_width * grid_width);
    
//     // 确保 n 至少为 1
//     let n = (n_samples as f32).sqrt().ceil() as u32;
//     if n == 0 {
//         // 这种情况不应该发生，因为我们已经确保 total_cancer_cells > 0
//         // 但作为额外的安全措施
//         let x = rng.gen_range(0..grid_width);
//         let y = rng.gen_range(0..grid_width);
//         let index = (y * grid_width + x) as usize;
//         if index < cell_grid.len() {
//             cell_grid[index] = 1; // 放置一个癌细胞
//         }
//         return;
//     }
    
//     // 跟踪已放置的采样点
//     let mut samples = Vec::new();
    
//     // 第一个采样点：随机放置在单位正方形内
//     let first_x = rng.gen_range(0.0..1.0);
//     let first_y = rng.gen_range(0.0..1.0);
//     samples.push((first_x, first_y));
    
//     if n_samples > 1 {
//         // 第二个采样点：放在对角相对的象限
//         let second_quadrant_x = if first_x < 0.5 { 1 } else { 0 };
//         let second_quadrant_y = if first_y < 0.5 { 1 } else { 0 };
        
//         // 在对角象限内随机位置
//         let second_x = second_quadrant_x as f32 * 0.5 + rng.gen_range(0.0..0.5);
//         let second_y = second_quadrant_y as f32 * 0.5 + rng.gen_range(0.0..0.5);
//         samples.push((second_x, second_y));
        
//         // 跟踪每行每列的占用情况 (N-rook rule)
//         let mut occupied_rows = vec![false; n as usize];
//         let mut occupied_cols = vec![false; n as usize];
        
//         // 标记前两个样本占用的行列
//         let row1 = (first_y * n as f32) as usize;
//         let col1 = (first_x * n as f32) as usize;
//         let row2 = (second_y * n as f32) as usize;
//         let col2 = (second_x * n as f32) as usize;
        
//         occupied_rows[row1] = true;
//         occupied_cols[col1] = true;
//         occupied_rows[row2] = true;
//         occupied_cols[col2] = true;
        
//         // 继续添加更多采样点，遵循 N-rook 规则
//         let mut current_level = 2; // 从4个象限开始，然后是16个，然后是64个...
        
//         while samples.len() < n_samples as usize {
//             let subdivisions = 1 << current_level; // 2^current_level
//             let cell_size = 1.0 / subdivisions as f32;
            
//             // 计算每个主要区域包含多少个子区域
//             // 确保不会除以零
//             let subdivs_per_major = if n == 0 { 1 } else { subdivisions as usize / n as usize };
//             if subdivs_per_major == 0 {
//                 // 如果细分级别太低，增加到下一级
//                 current_level += 1;
//                 continue;
//             }
            
//             // 为当前细分级别创建新的行/列占用跟踪
//             let mut level_occupied_rows = vec![vec![false; subdivs_per_major]; n as usize];
//             let mut level_occupied_cols = vec![vec![false; subdivs_per_major]; n as usize];
            
//             // 标记已有样本占用的行列
//             for &(x, y) in &samples {
//                 let row = (y * subdivisions as f32) as usize;
//                 let col = (x * subdivisions as f32) as usize;
                
//                 // 找到样本所在的具体单元格
//                 let major_row = row / subdivs_per_major;
//                 let major_col = col / subdivs_per_major;
                
//                 if major_row < level_occupied_rows.len() && major_col < level_occupied_cols.len() {
//                     let sub_row = row % subdivs_per_major;
//                     let sub_col = col % subdivs_per_major;
                    
//                     if sub_row < level_occupied_rows[major_row].len() && 
//                        sub_col < level_occupied_cols[major_col].len() {
//                         level_occupied_rows[major_row][sub_col] = true;
//                         level_occupied_cols[major_col][sub_row] = true;
//                     }
//                 }
//             }
            
//             // 尝试放置新样本
//             for i in 0..n as usize {
//                 for j in 0..n as usize {
//                     if samples.len() >= n_samples as usize {
//                         break;
//                     }
                    
//                     if !occupied_rows[i] || !occupied_cols[j] {
//                         // 找到可用的行/列组合
//                         let mut available_cells = Vec::new();
                        
//                         let row_start = i * subdivs_per_major;
//                         let row_end = ((i + 1) * subdivs_per_major).min(subdivisions as usize);
//                         let col_start = j * subdivs_per_major;
//                         let col_end = ((j + 1) * subdivs_per_major).min(subdivisions as usize);
                        
//                         for row in row_start..row_end {
//                             for col in col_start..col_end {
//                                 let sub_row = row - row_start;
//                                 let sub_col = col - col_start;
                                
//                                 if sub_row < level_occupied_rows[i].len() && 
//                                    sub_col < level_occupied_cols[j].len() && 
//                                    !level_occupied_rows[i][sub_col] && 
//                                    !level_occupied_cols[j][sub_row] {
//                                     available_cells.push((row, col));
//                                 }
//                             }
//                         }
                        
//                         if !available_cells.is_empty() {
//                             // 随机选择一个可用单元格
//                             let (row, col) = available_cells[rng.gen_range(0..available_cells.len())];
                            
//                             // 在单元格内随机放置
//                             let x = (col as f32 + rng.gen_range(0.0..1.0)) * cell_size;
//                             let y = (row as f32 + rng.gen_range(0.0..1.0)) * cell_size;
                            
//                             samples.push((x, y));
//                             occupied_rows[i] = true;
//                             occupied_cols[j] = true;
                            
//                             // 放置对角相对的样本（如果可能）
//                             if samples.len() < n_samples as usize && n >= 2 {
//                                 let diag_i = (i + n as usize / 2) % n as usize;
//                                 let diag_j = (j + n as usize / 2) % n as usize;
                                
//                                 if !occupied_rows[diag_i] && !occupied_cols[diag_j] {
//                                     let mut diag_available = Vec::new();
                                    
//                                     let diag_row_start = diag_i * subdivs_per_major;
//                                     let diag_row_end = ((diag_i + 1) * subdivs_per_major).min(subdivisions as usize);
//                                     let diag_col_start = diag_j * subdivs_per_major;
//                                     let diag_col_end = ((diag_j + 1) * subdivs_per_major).min(subdivisions as usize);
                                    
//                                     for diag_row in diag_row_start..diag_row_end {
//                                         for diag_col in diag_col_start..diag_col_end {
//                                             let sub_row = diag_row - diag_row_start;
//                                             let sub_col = diag_col - diag_col_start;
                                            
//                                             if sub_row < level_occupied_rows[diag_i].len() && 
//                                                sub_col < level_occupied_cols[diag_j].len() && 
//                                                !level_occupied_rows[diag_i][sub_col] && 
//                                                !level_occupied_cols[diag_j][sub_row] {
//                                                 diag_available.push((diag_row, diag_col));
//                                             }
//                                         }
//                                     }
                                    
//                                     if !diag_available.is_empty() {
//                                         let (diag_row, diag_col) = diag_available[rng.gen_range(0..diag_available.len())];
                                        
//                                         let diag_x = (diag_col as f32 + rng.gen_range(0.0..1.0)) * cell_size;
//                                         let diag_y = (diag_row as f32 + rng.gen_range(0.0..1.0)) * cell_size;
                                        
//                                         samples.push((diag_x, diag_y));
//                                         occupied_rows[diag_i] = true;
//                                         occupied_cols[diag_j] = true;
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
            
//             current_level += 1;
            
//             // 如果我们已经达到了最高细分级别但仍未放置足够的样本，放宽N-rook约束
//             if current_level > 10 {
//                 // 简单地在剩余空间中随机填充
//                 while samples.len() < n_samples as usize {
//                     let x = rng.gen_range(0.0..1.0);
//                     let y = rng.gen_range(0.0..1.0);
//                     samples.push((x, y));
//                 }
//             }
//         }
//     }
    
//     // 将生成的样本映射到网格上并放置癌细胞
//     for (x, y) in samples {
//         let grid_x = (x * (grid_width as f32 - 1.0)) as u32;
//         let grid_y = (y * (grid_width as f32 - 1.0)) as u32;
//         let index = (grid_y * grid_width + grid_x) as usize;
        
//         if index < cell_grid.len() && cell_grid[index] == 0 {  // 只在空白单元格放置癌细胞
//             cell_grid[index] = 1;  // 假设1表示癌细胞
//             cancer_cells_placed += 1;
            
//             if cancer_cells_placed >= total_cancer_cells {
//                 break;
//             }
//         }
//     }
    
//     // 如果还没放置足够的癌细胞，用随机方法填充剩余的
//     while cancer_cells_placed < total_cancer_cells {
//         let x = rng.gen_range(0..grid_width);
//         let y = rng.gen_range(0..grid_width);
//         let index = (y * grid_width + x) as usize;
        
//         if index < cell_grid.len() && cell_grid[index] == 0 {
//             cell_grid[index] = 1;  // 放置癌细胞
//             cancer_cells_placed += 1;
//         }
//     }
// }

pub fn random_init_cancer(cell_grid: &mut Vec<u32>, grid_width: u32, init_cancer_rate: f32) {
    let mut rng = rand::thread_rng();
    let total_cancer_cells = (init_cancer_rate * (grid_width * grid_width) as f32) as u32;
    for _ in 0..total_cancer_cells {
        let x = rng.gen_range(0..grid_width);
        let y = rng.gen_range(0..grid_width);
        let index = (y * grid_width + x) as usize;
        cell_grid[index] = 1;
    }
}