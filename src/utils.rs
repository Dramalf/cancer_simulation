pub fn count_value(data: &[u32], value: u32) -> usize {
    data.iter().filter(|&&x| x == value).count()
}