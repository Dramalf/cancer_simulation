use crate::status::Status;
pub fn status_log(status: &Status, time_stamp: u32) {
    let log_message = &format!(
        " Time {}
-----------------------------------
| Cell Type   │ Count  │ Percent   │
-----------------------------------
| Cancer      │ {:<6} │ {:>6.2}%   │
| Normal      │ {:<6} │ {:>6.2}%   │
| WBC         │ {:<6} │ {:>6.2}%   │
| Regenerated │ {:<6} │ {:>6.2}%   │
| Dead        │ {:<6} │ {:>6.2}%   │
| CTT         │ {:<6} │ {:>6.2}%   │
-----------------------------------",
        time_stamp,
        status.cancer_cells,
        status.cancer_percent,
        status.normal_cells,
        status.normal_percent,
        status.wbc_cells,
        status.wbc_percent,
        status.regenerated_cells,
        status.regenerated_percent,
        status.dead_cells,
        status.dead_percent,
        status.ctt,
        status.ctt_percent
    );
    println!("{}", log_message);
}
pub fn info_log(log_message: &str, time_stamp: u32) {
    println!("Time {} : {}", time_stamp, log_message);
}
