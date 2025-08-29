# Dynamic Policy Fusion for Reinforcement Learning Agents

This project implements the methodology described in the paper "Dynamic Policy Fusion for User Alignment Without Re-Interaction". It provides a framework for personalizing a pre-trained reinforcement learning (RL) agent's policy to align with a user's intent, without needing to retrain the agent from scratch.

The core idea is to fuse a **task-specific policy** (learned by a standard RL agent like DQN) with an **intent-specific policy** (inferred from user feedback via an LSTM). The fusion is done dynamically to ensure the agent accomplishes its primary task while also adhering to the user's preferences.

## Results

The grid below showcases the behavior of the agent under three different policies across various environments.
* **DQN**: The agent follows the base policy, which is trained to maximize the task reward only.
* **LSTM**: The agent follows the intent-only policy, which captures the human intent may not be optimal for the main task.
* **Fusion**: The agent uses the dynamic fusion policy, balancing task completion with user preferences.

| DQN | LSTM | Fusion |
| :---: | :---: | :---: |
| ![DQN Result](Results/Highway/dqn.gif) | ![LSTM Result](Results/Highway/lstm.gif) | ![Fusion Result](Results/Highway/fusion.gif) |
<!-- | ![DQN Result 2](Results/dqn_2.gif) | ![LSTM Result 2](Results/lstm_2.gif) | ![Fusion Result 2](Results/fusion_2.gif) |
| ![DQN Result 3](Results/dqn_3.gif) | ![LSTM Result 3](Results/lstm_3.gif) | ![Fusion Result 3](Results/fusion_3.gif) | -->


## How it Works

1.  **Task-Specific Agent**: A standard DQN agent is trained to master a task (e.g., win at Pong, navigate a highway). Its learned policy is focused purely on maximizing the task reward.
2.  **Intent-Specific Model**: An LSTM model is trained on trajectories labeled with user feedback. It learns to predict Q-values that represent the user's preferences (e.g., "prefer staying in the middle lane").
3.  **Dynamic Policy Fusion**: During evaluation, the script:
    * Gets Q-values from both the pre-trained DQN and the LSTM.
    * Converts these Q-values into probability distributions (policies) using a Boltzmann distribution.
    * Dynamically adjusts the "temperature" of the intent-specific policy based on how well the agent has been following the user's preferences so far. This prevents one policy from completely dominating the other.
    * Fuses the two policies by multiplying them together.
    * Selects the action with the highest probability from the final fused policy.

## File Structure

* `train_agent.py`: Script to train the task-specific DQN agent.
* `train_lstm.py`: Script to train the intent-specific LSTM model.
* `test_fusion.py`: The main script to evaluate the DQN, LSTM, and the dynamic fusion strategy.
* `configs/`: Directory containing configuration files managed by Hydra.
    * `test_fusion.yaml`: The main configuration file for evaluation.
    * Other `.yaml` files for agent, environment, and model parameters.
* `architectures/`: Contain basic arcitectures and utility functions.
    * `common_utils.py`: Should contain helper functions like `Boltzmann` and `save_gif`.
* `architectures/`: Contains the agent definition and agent utils
* `lstm/`: Contains the LSTM definition and agent utils
* `labeller/`: Contains the simulated labeller
* `env/`: Contains the environment implementations (e.g., `highway.py`, `pong.py`).

## Dependencies

This project uses:
* Python 3.x
* PyTorch
* NumPy
* Hydra (for configuration management)
* An environment library (e.g., `highway-env`, `pygame`)

## How to Run

The process is divided into three main steps: training the agent, training the intent model, and finally, evaluating the fusion strategy. This project uses [Hydra](https://hydra.cc/) to manage complex configurations.

### Step 1: Train the Task-Specific Agent

Run the following command to train the DQN agent. The trained model will be saved for the evaluation step.

```bash
python train_agent.py
```

### Step 2: Train the Intent-Specific Model

Next, train the LSTM on trajectories with user feedback.

```bash
python train_lstm.py
```

### Step 3: Evaluate the Fusion Strategy

Once both models are trained, you can run the evaluation.

**To run an evaluation with the fusion strategy:**

```bash
python test_fusion.py
```

This will run the experiment using the default parameters defined in `configs/test_fusion.yaml`, loading the pre-trained models.

**To change the evaluation strategy or other parameters:**

You can override any configuration parameter from the command line.

* **Evaluate with only the DQN agent:**
    ```bash
    python test_fusion.py strategy=dqn
    ```

* **Evaluate with only the LSTM model:**
    ```bash
    python test_fusion.py strategy=lstm
    ```


The results (e.g., GIFs of the episodes) will be saved to the directory specified by the `result_path` parameter in your configuration file.
