
# WGPU-based Cellular Automaton for Cancer Simulation

This project is a cancer transformation simulation program based on cellular automata. It simulates the interaction and evolution of different cell types, including normal cells, cancer cells, dead cells, and regenerated cells, on a two-dimensional grid. It also considers the response of the immune system and the cancer target therapy.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c543a1a7-8765-4c16-bae0-66847ed42246" alt="output" />
</p>

## Getting Started

### Prerequisites

You need to have the Rust programming environment installed. If you haven't installed it yet, you can follow the official guide.

* **Install Rust and Cargo**: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

This will install both the Rust compiler (`rustc`) and the package manager and build tool (`cargo`).

### Configuration

You can modify the simulation's parameters by editing the `config.json` file.

* **grid_width**: Width of the canvas.
* **num_frames**: Duration of the simulation; set to `0` for infinite simulation.
* **init_cancer_rate**: The amount of the initial cancer cells is calculated by $w^2\times r$.
* **init_strategy**: `progressive`,`random` or `grid`.
* **sleep_time**: Interval between each frame rendering.
* **cancer_transformation_prob**: The probability that a normal cell will be infected by its neighbor cancer cells.
* **init_cancer_grid_num**: If the `init_strategy` is `grid`, this controlls the number of initial cancer sub-grids.
* **init_cancer_grid_width**: If the `init_strategy` is `grid`, this controlls the side length of initial cancer sub-grids.
* **cell_regeneration_prob**: The probability of cell regeneration from the dead cell position.
* **regen_invincible_time**: The duration of immune invincible for regenerated cells.

**Note**: This feature is still under development, and not all parameters in the file are currently used in the simulation. For more advanced simulation settings, you may need to modify the source code.

### Running the Simulation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Dramalf/cancer_simulation
    cd cancer_simulation
    ```
2.  Run the simulation:
    ```bash
    cargo run
    ```

The program will start, and you will see the simulation window.

## User Guide

* **Start Simulation**: The simulation starts automatically when you run `cargo run`.
* **Pause/Menu**: Click the left mouse button in the window to pause the simulation and bring up the operations menu.
* **Debug**: Select Debug mode and click the position in the grid to log the details of the cell grid.
* **Visual Representation**:
    * **Flesh**: Normal Cells
    * **Red**: Cancer Cells
    * **Black**: Dead Cells
    * **Green**: Regenerated Cells
    * **Pink**: As the immune system responds and fights cancer cells, areas with higher concentrations will appear pinker.
* **Targeted Therapy**: The targeted therapy functionality is not yet enabled.
* **Data Export**: You can save the simulation data to a CSV file through the menu for further analysis with the provided `plot.py` script.

##  Interesting insight

The spread of cancer cells follows a logistic growth pattern. Early merging of small tumors into a larger one leads to a decrease in the overall propagation speed (though physiological factors related to tumor size are not considered in this project).

![image](https://github.com/user-attachments/assets/7df8c974-c31a-41bc-bf30-7e037ebf8082)

![image](https://github.com/user-attachments/assets/a42afe6b-264d-4a29-b98f-bfc94a3273f2)

