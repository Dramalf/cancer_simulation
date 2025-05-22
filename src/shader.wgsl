//shader.wgsl

struct View {
    output_resolution : vec2 < u32>,
    grid_resolution : vec2 <u32>,
};
struct CellParams{
    cancer_transformation_prob : f32,
    cell_regeneration_prob : f32,
    wbc_degeneration_prob : f32,
    wbc_regeneration_prob : f32,
    time_stamp : u32,
    regen_invincible_time : u32,
    ctt_effect : u32,
}

const NORMAL_CELL = 0u;
const CANCER_CELL = 1u;
const DEAD_CELL = 2u;
const REGENERATED_CELL = 3u;
const WBC = 4u;
const CTT = 8u;

@group(0) @binding(0) var<storage, read_write> grid_data : array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_data_prev : array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_data_curr : array<u32>;
@group(0) @binding(3) var<storage, read_write> wbc_data_prev : array<f32>;
@group(0) @binding(4) var<storage, read_write> wbc_data_curr : array<f32>;
@group(0) @binding(5) var<storage, read_write> ctt_data_prev : array<f32>;
@group(0) @binding(6) var<storage, read_write> ctt_data_curr : array<f32>;
@group(0) @binding(7) var<uniform> view : View;
@group(0) @binding(8) var<uniform> cell_params : CellParams;
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4 < f32> {
    //生成覆盖整个裁剪空间的两个三角形 (形成一个正方形)
    //顶点顺序: 左下, 右下, 左上, 左上, 右下, 右上
    var positions = array<vec2 < f32>, 6 > (
    vec2 < f32 > (-1.0, -1.0),
    vec2 < f32 > (1.0, -1.0),
    vec2 < f32 > (-1.0, 1.0),
    vec2 < f32 > (-1.0, 1.0),
    vec2 < f32 > (1.0, -1.0),
    vec2 < f32 > (1.0, 1.0) //右上
    );
    return vec4 < f32 > (positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord : vec4 < f32>) -> @location(0) vec4 < f32> {
    //将片段坐标从像素坐标转换为网格单元坐标
    //frag_coord.x ranges from 0.5 to output_resolution.x - 0.5
    //frag_coord.y ranges from 0.5 to output_resolution.y - 0.5

    //计算当前片段对应于哪个网格单元
    let cell_f32_x = (frag_coord.x / f32(view.output_resolution.x)) * f32(view.grid_resolution.x);
    let cell_f32_y = (frag_coord.y / f32(view.output_resolution.y)) * f32(view.grid_resolution.y);

    let grid_x = u32(floor(cell_f32_x));
    let grid_y = u32(floor(cell_f32_y));

    //边界检查 (理论上如果窗口尺寸是网格尺寸的整数倍，且frag_coord正确，可以不严格需要)
    if (grid_x >= view.grid_resolution.x || grid_y >= view.grid_resolution.y)
    {
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }

    let index = grid_y * view.grid_resolution.x + grid_x;
    let value = grid_data[index];
    if (value == NORMAL_CELL)
    {
        return vec4 < f32 > (0.93, 0.8, 0.69, 1.0);
    } else if (value == CANCER_CELL)
    {
        return vec4 < f32 > (1.0, 0.0, 0.0, 1.0);
    } else if (value == DEAD_CELL)
    {
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }else if (value == REGENERATED_CELL)
    {
        return vec4 < f32 > (0.0, 1.0, 0.0, 1.0);
    }
    else if(value >= WBC && value < CTT)
    {
        return vec4 < f32 > (0.0, 0.0, 1.0, 1.0);
    }else if(value >= CTT)
    {
        return vec4 < f32 > (0.5 + f32(value - cell_params.ctt_effect) / f32(cell_params.ctt_effect), 0.2, 0.4, 1.0);
    }else{
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }
}

fn get_v(x : u32, y : u32) -> u32 {
    let index = x * view.grid_resolution.x + y;
    if x < view.grid_resolution.x && x >= 0u && y < view.grid_resolution.y && y >= 0u {
        return grid_data[index];
    }
    return 0u;
}
fn get_cell_v_prev(x : u32, y : u32) -> u32 {
    let index = x * view.grid_resolution.x + y;
    if x < view.grid_resolution.x && x >= 0u && y < view.grid_resolution.y && y >= 0u {
        return cell_data_prev[index];
    }
    return 0u;
}
fn get_wbc_v_prev(x : u32, y : u32) -> f32 {
    let index = x * view.grid_resolution.x + y;
    if x < view.grid_resolution.x && x >= 0u && y < view.grid_resolution.y && y >= 0u {
        return wbc_data_prev[index];
    }
    return 0u;
}
fn get_ctt_v_prev(x : u32, y : u32) -> f32 {
    let index = x * view.grid_resolution.x + y;
    if x < view.grid_resolution.x && x >= 0u && y < view.grid_resolution.y && y >= 0u {
        return ctt_data_prev[index];
    }
    return 0u;
}
fn count_wbc_neighbors(x : u32, y : u32) -> f32 {
    let neighbors : array<vec2 < u32>, 4> = array<vec2 < u32>, 4 > (
    vec2 < u32 > (x + 1u, y),
    vec2 < u32 > (x - 1u, y),
    vec2 < u32 > (x, y + 1u),
    vec2 < u32 > (x, y - 1u)
    );

    var count : u32 = 0.0f;
    for (var j : u32 = 0u; j < 4u; j = j + 1u)
    {
        count += get_wbc_v_prev(neighbors[j].x, neighbors[j].y);
    }
    return count;
}

fn count_cancer_neighbors(x : u32, y : u32) -> u32 {
    let neighbors = array<vec2u, 4 > (
    vec2u(x + 1, y),
    vec2u(x - 1, y),
    vec2u(x, y + 1),
    vec2u(x, y - 1),
    );
    var count : u32 = 0u;
    var i : u32 = 0u;
    for (i = 0u; i < 4u; i++)
    {
        let neighbor_value = get_cell_v_prev(neighbors[i].x, neighbors[i].y);
        if neighbor_value == CANCER_CELL {
            count += 1u;
        }
    }
    return count;

}

fn count_CTT_neighbors(x : u32, y : u32) -> vec2u {
    let neighbors = array<vec2u, 4 > (
    vec2u(x + 1, y),
    vec2u(x - 1, y),
    vec2u(x, y + 1),
    vec2u(x, y - 1),
    );
    var count : u32 = 0u;
    var age : u32 = 0u;
    var i : u32 = 0u;
    for (i = 0u; i < 4u; i++)
    {
        let neighbor_value = get_ctt_v_prev(neighbors[i].x, neighbors[i].y);
        if neighbor_value >= CTT {
            count += 1u;
            age = max(age, neighbor_value);
        }
    }
    return vec2u(count, age);

}




@compute @workgroup_size(1)
fn cancer_transformation(@builtin(global_invocation_id) global_id : vec3u)
{
    let x = global_id.x / view.grid_resolution.x;
    let y = global_id.x % view.grid_resolution.y;

    //draw last frame
    grid_data[global_id.x] = cell_data_prev[global_id.x];

    //count immune neighbors
    let wbc_count = count_wbc_neighbors(x, y);
    let ctt_info = count_CTT_neighbors(x, y);
    let cct_count = ctt_info.x;
    let cct_age = ctt_info.y;

    //If there is immune cell, cover the cell by immune cell
    if wbc_data_prev[global_id.x] != 0u {
        grid_data[global_id.x] = wbc_data_prev[global_id.x];
    }
    if ctt_data_prev[global_id.x] != 0u {
        grid_data[global_id.x] = ctt_data_prev[global_id.x];
    }

    let prev_cell = get_cell_v_prev(x, y);
    var next_cell = 0u;
    //The cancer cell will be killed by immune cell with probability 0.55
    //TODO: Make the probability configurable and related to the number of immune cells
    if prev_cell == CANCER_CELL {
        if wbc_count >= 1.0f {
            next_cell = select(DEAD_CELL, CANCER_CELL, pcg_2u_3f(vec2u(global_id.x, cell_params.time_stamp)).x < 0.55);
        }
    }
    //The dead cell will be regenerated with probability of cell_regeneration_prob
    else if prev_cell == DEAD_CELL {
        let input_seed = vec2u(global_id.x, cell_params.time_stamp);
        let output_seed = pcg_2u_3f(input_seed);
        next_cell = select(DEAD_CELL, REGENERATED_CELL, output_seed.x < cell_params.cell_regeneration_prob);
    }
    else if prev_cell == REGENERATED_CELL {
        //The regenerated cell has a regen_invincible_time immune period, scale by 2 since E(E(life since t)*P(regen at t)) = inv_time/2;
        if cell_params.time_stamp % (2 * cell_params.regen_invincible_time) == 0u {
            next_cell = NORMAL_CELL;
        }
    }
    //Assume the wbc move to the up/down/left/right cell with probability 0.25
    //白细胞更新
    wbc_data_curr[global_id.x] = get_wbc_v_prev(x, y) * 0.75f + 0.25f * wbc_count;

    //cancer cell transformation
    if next_cell == NORMAL_CELL{
        if cancer_count == 4u{
            next_cell = DEAD_CELL;
        }
        else{
            let not_transform_prob = pow(1.0f - cell_params.cancer_transformation_prob, f32(cancer_count));
            next_cell = select(CANCER_CELL, NORMAL_CELL, random(x, y, cancer_count * cell_params.time_stamp) < not_transform_prob);
        }
    }


    //白细胞离开
    //else if CTT > current_cell && current_cell >= WBC {
    //let prev_cell = current_cell % WBC;
    //let input_seed = vec2u(cell_params.time_stamp, current_cell);
    //let output_seed = pcg_2u_3f(input_seed);
    //grid_data[global_id.x] = select(prev_cell, current_cell, output_seed.x < 0.25);
    //}
    ////靶向药扩散/药效减弱
    //else if current_cell >= CTT {
    //if current_cell == CTT {
    //grid_data[global_id.x] = REGENERATED_CELL;
    //}
    //else{
    //grid_data[global_id.x] = current_cell - 1u;
    //}
    //}

    //癌细胞扩散
    //if grid_data[global_id.x] == NORMAL_CELL {
    //let cancer_count = count_cancer_neighbors(x, y);
    //if cancer_count >= 1u && cancer_count <= 3u {
    //let input_seed = vec2u(global_id.x, cell_params.time_stamp + cancer_count);
    //let output_seed = pcg_2u_3f(input_seed);
    //grid_data[global_id.x] = select(CANCER_CELL, NORMAL_CELL, output_seed.x < pow(1.0f - cell_params.cancer_transformation_prob, f32(cancer_count)));
    //if grid_data[global_id.x] == CANCER_CELL && cct_count >= 1u {
    //grid_data[global_id.x] = select(CANCER_CELL, cct_age, pseudo_random(x, y, cell_params.time_stamp) < 0.05f);
    //}
    //} else if cancer_count == 4u {
    //grid_data[global_id.x] = 2u;
    //}
    //}

    //靶向药移动
    //if cct_count >= 1u && grid_data[global_id.x] == CANCER_CELL{
    //let input_seed = vec2u(x * y+cct_count, cell_params.time_stamp);
    //let output_seed = pcg_2u_3f(input_seed);
    //let cct_prob = clamp(f32(cct_age) / f32(cell_params.ctt_effect) * 0.25, 0.0f, 0.05f);
    //grid_data[global_id.x] = select(CANCER_CELL, cct_age - 1u, output_seed.x < cct_prob);
    //}
}



    //https://github.com/mighdoll/random-wgsl/blob/main/src/lib.wgsl
    //PCG pseudo random generator from vec2u to vec4f
    //the random output is in the range from zero to 1
fn pcg_2u_3f(pos : vec2u) -> vec3f {
    let seed = mix2to3(pos);
    let random = pcg_3u_3u(seed);
    let normalized = ldexp(vec3f(random), vec3(-32));
    return vec3f(normalized);
}

    //PCG random generator from vec3u to vec3u
    //adapted from http://www.jcgt.org/published/0009/03/02/
fn pcg_3u_3u(seed : vec3u) -> vec3u {
    var v = seed * 1664525u + 1013904223u;

    v = mixing(v);
    v ^= v >> vec3(16u);
    v = mixing(v);

    return v;
}

    //permuted lcg
fn mixing(v : vec3u) -> vec3u {
    var m : vec3u = v;
    m.x += v.y * v.z;
    m.y += v.z * v.x;
    m.z += v.x * v.y;

    return m;
}

    //mix position into a seed as per: https://www.shadertoy.com/view/XlGcRh
fn mix2to3(p : vec2u) -> vec3u {
    let seed = vec3u(
    p.x,
    p.x ^ p.y,
    p.x + p.y,
    );
    return seed;
}

    //from https://stackoverflow.com/questions/12964279/whats-the-origin-of-this-glsl-rand-one-liner
fn sinRand(co : vec2f) -> f32 {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

fn pseudo_random(x : u32, y : u32, tick : u32) -> f32 {
    let seed = x * 374761393 + y * 668265263 + tick * 982451653;
    let hashed = seed ^ (seed >> 13);
    return fract(sin(f32(hashed)) * 43758.5453);
}

fn random(x : u32, y : u32, seed : u32) -> f32 {
    var h = x * 374761393u + y * 668265263u + seed * 144066263297769815u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    let r = h ^ (h >> 16u);
    return f32(r & 0xFFFFu) / 65535.0;
}
