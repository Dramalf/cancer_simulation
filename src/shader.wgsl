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
    else if (value >= 80u)
    {
        let c = f32(value) / 255.0f;
        return vec4 < f32 > (1.0, c, c, 1.0);
    }
    //else if(value >= WBC && value < CTT)
    //{
    //return vec4 < f32 > (0.0, 0.0, 1.0, 1.0);
    //}else if(value >= CTT)
    //{
    //return vec4 < f32 > (0.5 + f32(value - cell_params.ctt_effect) / f32(cell_params.ctt_effect), 0.2, 0.4, 1.0);
    //}
    else{
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
    return 0.0f;
}
fn get_ctt_v_prev(x : u32, y : u32) -> f32 {
    let index = x * view.grid_resolution.x + y;
    if x < view.grid_resolution.x && x >= 0u && y < view.grid_resolution.y && y >= 0u {
        return ctt_data_prev[index];
    }
    return 0.0f;
}
fn get_wbc_concentration(x : u32, y : u32) -> f32 {
    let neighbors : array<vec2 < u32>, 4> = array<vec2 < u32>, 4 > (
    vec2 < u32 > (x + 1u, y),
    vec2 < u32 > (x - 1u, y),
    vec2 < u32 > (x, y + 1u),
    vec2 < u32 > (x, y - 1u)
    );

    var count : f32 = 0.0f;
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
        //let neighbor_value = get_ctt_v_prev(neighbors[i].x, neighbors[i].y);
        //if neighbor_value >= CTT {
        //count += 1u;
        //age = max(age, neighbor_value);
        //}
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
    let wbc_concentration = get_wbc_concentration(x, y);
    //let ctt_info = count_CTT_neighbors(x, y);
    //let cct_count = ctt_info.x;
    //let cct_age = ctt_info.y;



    let prev_cell = get_cell_v_prev(x, y);
    var next_cell = prev_cell;
    //If there is immune cell, cover the cell by immune cell
    if wbc_concentration >= 0.01 && prev_cell == CANCER_CELL{
        grid_data[global_id.x] = clamp(u32(wbc_concentration * 80.0f) + 80u, 0u, 255u);
    }
    if ctt_data_prev[global_id.x] != 0.0f {
        grid_data[global_id.x] = CTT;
    }
    //The cancer cell will be killed by immune cell with probability 0.55
    //TODO: Make the probability configurable and related to the number of immune cells
    if prev_cell == CANCER_CELL {
        if wbc_concentration> 0.01{
            let input_seed = vec2u(global_id.x, cell_params.time_stamp);
            let output_seed = pcg_2u_3f(input_seed);
            let wbc_effect = effect_hill(wbc_concentration, 0.8, 1.0, 0.45);
            next_cell = select(DEAD_CELL, CANCER_CELL, output_seed.x > wbc_effect);
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
    }else if prev_cell == NORMAL_CELL{
        //cancer cell transformation

        let cancer_count = count_cancer_neighbors(x, y);
        if cancer_count == 4u{
            next_cell = DEAD_CELL;
        }
        else if cancer_count >= 1u && cancer_count <= 3u{
            let not_transform_prob = pow(1.0f - cell_params.cancer_transformation_prob, f32(cancer_count));
            let input_seed = vec2u(global_id.x, cell_params.time_stamp);
            let output_seed = pcg_2u_3f(input_seed);
            next_cell = select(CANCER_CELL, NORMAL_CELL, output_seed.x < not_transform_prob);
        }
    }

    cell_data_curr[global_id.x] = next_cell;
        //Assume the wbc move to the up/down/left/right cell with probability 0.25
    //白细胞更新
    wbc_data_curr[global_id.x] = 0.9f * 0.25f * wbc_concentration;

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

fn random(x : u32, y : u32, time : u32) -> f32 {
    var seed = x * 374761393u + y * 668265263u + time * 982451653u;
    seed = (seed ^ (seed >> 13u)) * 1274126177u;
    seed = seed ^ (seed >> 16u);
    return f32(seed & 0xFFFFu) / 65535.0;
}

//Bob Jenkins' One-At-A-Time hashing algorithm (adapted from GLSL to WGSL)
fn hash_bjoaat(x_in : u32) -> u32 {
    var x = x_in;
    x = x + (x << 10u);
    x = x ^ (x >> 6u);
    x = x + (x << 3u);
    x = x ^ (x >> 11u);
    x = x + (x << 15u);
    return x;
}

//Compound versions of the hashing algorithm
fn hash_bjoaat_vec2(v : vec2u) -> u32 {
    return hash_bjoaat(v.x ^ hash_bjoaat(v.y));
}

fn hash_bjoaat_vec3(v : vec3u) -> u32 {
    return hash_bjoaat(v.x ^ hash_bjoaat(v.y) ^ hash_bjoaat(v.z));
}

fn hash_bjoaat_vec4(v : vec4u) -> u32 {
    return hash_bjoaat(v.x ^ hash_bjoaat(v.y) ^ hash_bjoaat(v.z) ^ hash_bjoaat(v.w));
}

//Construct a float with half-open range [0:1] using low 23 bits.
//All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
fn float_construct_bjoaat(m_in : u32) -> f32 {
    let ieeeMantissa = 0x007FFFFFu; //binary32 mantissa bitmask
    let ieeeOne = 0x3F800000u;      //1.0 in IEEE binary32

    var m = m_in;
    m = m & ieeeMantissa;       //Keep only mantissa bits (fractional part)
    m = m | ieeeOne;            //Add fractional part to 1.0

    let f = bitcast < f32 > (m);//Range [1:2] in WGSL (equivalent to uintBitsToFloat)
    return f - 1.0;             //Range [0:1]
}

//Pseudo-random value in half-open range [0:1].
//WGSL does not have direct floatBitsToUint for vectors, so we process components.
//We'll provide a version for vec3f as that seems most relevant to your x,y,time input.

fn random_bjoaat_f32(x : f32) -> f32 {
    return float_construct_bjoaat(hash_bjoaat(bitcast < u32 > (x)));
}

fn random_bjoaat_vec2f(v : vec2f) -> f32 {
    //Hash components individually and combine, or hash a combined u32 representation
    let u_v = vec2u(bitcast < u32 > (v.x), bitcast < u32 > (v.y));
    return float_construct_bjoaat(hash_bjoaat_vec2(u_v));
}

fn random_bjoaat_vec3f(v : vec3f) -> f32 {
    let u_v = vec3u(bitcast < u32 > (v.x), bitcast < u32 > (v.y), bitcast < u32 > (v.z));
    return float_construct_bjoaat(hash_bjoaat_vec3(u_v));
}

fn random_bjoaat_vec4f(v : vec4f) -> f32 {
    let u_v = vec4u(bitcast < u32 > (v.x), bitcast < u32 > (v.y), bitcast < u32 > (v.z), bitcast < u32 > (v.w));
    return float_construct_bjoaat(hash_bjoaat_vec4(u_v));
}

//Example of how you might use it with x, y, time
fn get_random_f32_bjoaat(x_coord : u32, y_coord : u32, time_val : u32) -> f32 {
    //Convert u32 coordinates to f32 for hashing, or adapt hash to take mixed types
    //For simplicity, let's cast u32 to f32. Better might be to use their u32 bits directly.
    let inputs = vec3f(f32(x_coord), f32(y_coord), f32(time_val));
    return random_bjoaat_vec3f(inputs);
}

fn effect_hill(c : f32, half_effect_c : f32, hill_coefficient : f32, max_effect : f32) -> f32{
    return max_effect * pow(c, hill_coefficient) / (pow(c, hill_coefficient) + pow(half_effect_c, hill_coefficient));
}
