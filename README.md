# DeepParkour

### Training a humanoid agent to efficiently apply parkour skills to an obstacle environment.
## Dependencies
- requirements.txt includes all the dependecies required to run this project.
## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/aagrawal20/DeepParkour.git
    cd DeepParkour
    ```
- If have access to a CUDA-compatible gpu then install tensorflow gpu.
    ```bash 
    pip install tensorflow-gpu 
    ```
     Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install DeepParkour package.
    ```bash
    pip install -e .
    ```
## Training Agent
- You can train an agent using the train_agent.py file.
- You can add specific flags to the argument parser. 
    ```bash
    python src/main/train_agent.py
    ```
## Rendering an Agent
- You can render an agent using the render_agent.py file.
- You can add specific flags to the argument parser.
    ```bash
    python src/main/render_agent.py
    ```
    Note: PyBullet only supports CPU rendering. Turn off render flag or manually turn off gpu.

## Visualizing Agent training
- You can visualize different stastics for eg: loss vs timesteps or reward vs timesteps.
    ```bash
    jupyter notebook
    ```
- After loading the local host navigate to the visualization.ipynb in src/util.