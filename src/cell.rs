use rand::Rng;
use rand::prelude::*;

use pmj::{generate};
use rand::rngs::SmallRng;

pub enum CellType {
    NormalCell = 0,
    CancerCell = 1,
    DeadCell = 2,
    RegeneratedCell = 3,
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