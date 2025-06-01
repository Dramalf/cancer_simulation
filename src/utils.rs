pub fn count_value(data: &[u32], value: u32) -> usize {
    data.iter().filter(|&&x| x == value).count()
}

pub fn effect_hill(c:f32,half_effect_c:f32,hill_coefficient:f32) -> f32{
    return c.powf(hill_coefficient)/(c.powf(hill_coefficient)+half_effect_c.powf(hill_coefficient));
}