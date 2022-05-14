## LuxAI 

Spring 22 - CS394 - Reinforcement Learning Course Project

This is a fork of the top solution from the Lux AI 2021 competition, with modifications to improve and analyze the model. 

### Getting Started
To star training run the following command under luxai directory. 

``` sh
python3 run_monobeast.py
```


#### Change Hyperparameters
You can tune the hyperparameters through creating a config file in the config direcotry. Then change the config_name to the corresponding config file in run_monobeast.py.
Ex:

```
@hydra.main(config_path="conf", config_name="conv_phase4_small_teacher.yaml")
```

#### Run Tournament 
To run a tournament you have to use download the luxAI official repositry: https://github.com/Lux-AI-Challenge/Lux-Design-2021
Then in the models folder run: 
``` sh
lux-ai-2021 --tournament
```

#### Visualize game
All the replays would be stored at luxai/replay.
You could watch the replay by uploading the json file to: https://2021vis.lux-ai.org/. 


### Important Folders and Files
conf:  This directory store all hyperparameter setting for the training.
Imitation_learning - 
Lux AI: The game environment and the neural network block strucutres. In 'luxai/lux_ai/'lux_gam' you can find the observation space, reward space, and rewards that we desgined.
Imitation_learning: Imitation learning approch that is build on Ironbar team's implementation
Internal_testing: public agents and our agents 
run_monobesat.py:  Start the training process. Includes the neural network archtecture. 


### Acknowledgements 
Our model is build on top these two implementations:
[Toad Brigade's DRL approach on LuxAI](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)

[Ironbar's Imitation Learning approach on LuxAI](https://github.com/ironbar/luxai)






