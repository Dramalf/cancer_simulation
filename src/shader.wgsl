//shader.wgsl

struct Uniforms {
    output_resolution : vec2 < u32>,
    grid_resolution : vec2 <u32>,
};
struct CellParams{
    cancer_transformation_prob : f32,
    cell_regeneration_prob : f32,
    wbc_degeneration_prob : f32,
    wbc_regeneration_prob : f32,
    time_stamp : u32,
}

@group(0) @binding(0) var<storage, read_write> grid_data : array<u32>; //u32 存储 0 或 1
@group(0) @binding(1) var<uniform> ubo : Uniforms;
@group(0) @binding(2) var<uniform> cell_params : CellParams;
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
    let cell_f32_x = (frag_coord.x / f32(ubo.output_resolution.x)) * f32(ubo.grid_resolution.x);
    let cell_f32_y = (frag_coord.y / f32(ubo.output_resolution.y)) * f32(ubo.grid_resolution.y);

    let grid_x = u32(floor(cell_f32_x));
    let grid_y = u32(floor(cell_f32_y));

    //边界检查 (理论上如果窗口尺寸是网格尺寸的整数倍，且frag_coord正确，可以不严格需要)
    if (grid_x >= ubo.grid_resolution.x || grid_y >= ubo.grid_resolution.y)
    {
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }

    let index = grid_y * ubo.grid_resolution.x + grid_x;
    let value = grid_data[index];
    if (value == 0u)
    {
        return vec4 < f32 > (1.0, 1.0, 1.0, 1.0);
    } else if (value == 1u)
    {
        return vec4 < f32 > (1.0, 0.0, 0.0, 1.0);
    } else if (value == 2u)
    {
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }else if (value == 3u)
    {
        return vec4 < f32 > (0.0, 1.0, 0.0, 1.0);
    }
    else if(value>3u && value<9u)
    {
        return vec4 < f32 > (0.0, 0.0, 1.0, 1.0);
    }else if(value>=9u){
        return vec4 < f32 > (0.5+f32(value-9u)/11.0, 1.0, 1.0, 1.0);
    }else{
        return vec4 < f32 > (0.0, 0.0, 0.0, 1.0);
    }
}

fn get_v(x : u32, y : u32) -> u32 {
    let index = x * ubo.grid_resolution.x + y;
    if x < ubo.grid_resolution.x && x >= 0u && y < ubo.grid_resolution.y && y >= 0u {
        return grid_data[index];
    }
    return 0u;
}

@compute @workgroup_size(1)
fn cancer_transformation(
@builtin(global_invocation_id)
global_id : vec3u,
)
{
    let x = global_id.x / ubo.grid_resolution.x;
    let y = global_id.x % ubo.grid_resolution.y;
    let current_cell = get_v(x, y);


    let wbc_count = count_wbc_neighbors(x, y);

    //癌细胞被杀死
    if current_cell == 1u {
        if wbc_count >=1u {
            grid_data[global_id.x] = 2u;
        }
    }
    //白细胞移动
    if current_cell >= 4u {
        let prev_cell = current_cell%4;
        grid_data[global_id.x] = prev_cell;
    }
    if current_cell < 4u && wbc_count >= 1u {
        let input_seed = vec2u(cell_params.time_stamp, global_id.x);
        let output_seed = pcg_2u_3f(input_seed);
        if current_cell == 1u {
            grid_data[global_id.x] = select(1u, 7u, output_seed.x < 0.55);

        }else{
        grid_data[global_id.x] = select(current_cell,current_cell+4u, output_seed.x < 0.35);

        }
    }
    //细胞再生
    if current_cell == 2u {
        let input_seed = vec2u(global_id.x, cell_params.time_stamp);
        let output_seed = pcg_2u_3f(input_seed);
        grid_data[global_id.x] = select(2u, 3u, output_seed.x < cell_params.cell_regeneration_prob);
    }

    //癌细胞扩散
    if get_v(x, y) == 0u {
        let cancer_count = count_cancer_neighbors(x, y);
        if cancer_count >= 1u && cancer_count <= 3u {
            let input_seed = vec2u(global_id.x, cell_params.time_stamp + cancer_count);
            let output_seed = pcg_2u_3f(input_seed);
            grid_data[global_id.x] =select(1u, 0u, output_seed.x < pow(1.0f - cell_params.cancer_transformation_prob, f32(cancer_count)));
        }else if cancer_count == 4u {
            grid_data[global_id.x] = 2u;
        }
    }
    if current_cell == 3u {
        grid_data[global_id.x] = 0u;
    }
}

//fn effect_to_kill_rate(effect: u32) -> f32 {
  //   if effect > 9u {
    //    return 0.1 + 0.9 * (f32(effect- 9u) / 11.0);
    //} else {
     //   return 0.1f;
    //}
//}

fn count_wbc_neighbors(x : u32, y : u32) -> u32 {
    let neighbors = array<vec2u, 4 > (
    vec2u(x + 1, y),
    vec2u(x - 1, y),
    vec2u(x, y + 1),
    vec2u(x, y - 1),
    );
    var count : u32 = 0u;
    var j : u32 = 0u;
    for (j = 0u; j < 4u; j++)
    {
        if get_v(neighbors[j].x, neighbors[j].y) >= 4u {
            count += 1u;
        }
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
        if get_v(neighbors[i].x, neighbors[i].y) == 1u {
            count += 1u;
        }
    }
    return count;

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

fn pseudo_random(x: u32, y: u32, tick: u32) -> f32 {
    let seed = x * 374761393 + y * 668265263 + tick * 982451653;
    let hashed = seed ^ (seed >> 13);
    return fract(sin(f32(hashed)) * 43758.5453);
}