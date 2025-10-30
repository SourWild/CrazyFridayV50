# Group4 Fencing 2 — MuJoCo-Gym Environment Guide

This project implements a MuJoCo-based fencing scenario environment (Gymnasium API), including a custom environment `ObstacleEnv`, a numerical IK and joint-position PD controller, a demo script, and logging. This README provides dependency setup, simulator notes, run instructions, and reproducibility steps.
 
## 1. Project Structure
- `obstacle_env.py`: Custom Gymnasium environment (`reset/step`) with event/termination logic and IK+PD control
- `ik_controller.py`: Numerical IK (finite-difference Jacobian + damped pseudoinverse) and joint-position PD controller
- `main.py`: Demo script (render/run steps/save CSV)
- `Fencing_agent&obstacle_description/`: MuJoCo XML and assets
- `simulation_output.csv`: Example output (time, qpos, ctrl) 
 
## 2. Dependencies & External Simulator
This project depends on MuJoCo and common Python scientific/RL libraries.

## 3. Create and Activate Environment (one-liner)
From project root:
```bash
conda env create -f environment.yml
conda activate drl_mujoco
```

## 4. Run Demo (copy-paste commands)
```bash
cd Group4_fencing_3
conda activate drl_mujoco
# Default: human render, reactive obstacle; adjust parameters at top of main.py if needed
python main.py
```
The run will produce `simulation_output.csv` containing per-step `qpos_*` and `ctrl_*` fields.
 
## 5. Key Parameter Settings
Defaults (modifiable at the top of `main.py`):
- Render mode: `render_mode = "human"`
- Obstacle mode: `obstacle_mode = 'reactive'` (options: `'dynamic'|'periodic'|'static'|'reactive'`)
- Steps / dt / episodes: `n_steps`, `dt`, `n_episodes`

## 6. TL;DR for Grading (Minimal Steps)
```bash
cd Group4_fencing_3
conda env create -f environment.yml
conda activate drl_mujoco
# Local GUI:
python main.py
# Or headless:
xvfb-run -a python main.py
```
This should reproduce on a standard Linux + Conda setup.