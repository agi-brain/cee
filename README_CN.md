# Reducing Action Space for Deep Reinforcement Learning via Causal Effect Estimation


## Requirements
We test our method with a gpu that can run CUDA 12.0. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```
source activate cee
```
## Instructions 
### Pre-training: Training N-value network
First, we need to train an N-value network. For example, in the Unlock Pickup environment, run:
```
cd cee
python min_red/grid_search_minigrid.py
```

### Phase 2: Conduct task training
When the pre-training is complete, add the model to ***makppo/train.py/Env_mask_dict***```
```
python maskppo/grid_search_minigrid.py
```

Both of two phases  will produce 'log' folder, where all the outputs are going to be stored. The data and lines can be observed in tensorboard.


```
tensorboard --logdir log
```
Besides, Operation of the PurePPO algorithm:

```
python pureppo/grid_search_minigrid.py
```

## 下面是详细步骤
### 预训练阶段
该阶段所有的运行文件全部在min_red文件夹中，可运行的文件有四个，分别对应四种不同环境，atari.py、grid_search_babyai.py、grid_search_maze.py、grid_search_minigrid.py，参数配置放在了min_red/config中，文件名就是环境名称，运行完毕后可以得到N值网络模型。
##### minigrid 环境
```
cd cee
python min_red/grid_search_minigrid.py
python min_red/grid_search_babyai.py
```

##### maze 环境
```
cd cee
python min_red/grid_search_maze.py
```
##### atari 环境
```
cd cee
python min_red/atari.py
```
### 第二阶段
我们把预训练阶段的到的模型，加在第二阶段中，即在maskppo/train.py中的30行，按照已有格式加进去，然后运行即可。
##### minigrid 环境
```
cd cee
python maskppo/grid_search_minigrid.py
python maskppo/grid_search_babyai.py
```
若想要运行环境中不同的任务，只需要在对应的py文件中修改即可，比如在grid_search_babyai.py中，将该py文件中15行修改想要运行的环境名称即可（GoToR3GreyKeyAddPositionBonus-v0） 
##### maze 环境
```
cd cee
python maskppo/grid_search_maze.py
```
##### atari 环境
```
cd cee
python maskppo/grid_atari.py
```


首先对于我们的五种方法，在maskppo/train.py中，已经将图中的四种曲线方法设置好，分别为cee、npm、cee-woc、npm_random。只需要将第115行中的参数改成对应的方法即可运行（此处运行生成的文件名称中没有设置方法名，如140行，需要注意一下），参数配置在config中。

注：除了PPO算法在单独的文件夹中，即pureppo文件夹中，主函数为pureppo/train.py，标准PPO算法不需要运行预训练阶段，不同环境的参数配置放在了pureppo/config中yaml文件，文件名对应环境名。
##### minigrid 环境
```
cd cee
python pureppo/grid_search_minigrid.py
python pureppo/grid_search_babyai.py
```


##### maze 环境
```
cd cee
python pureppo/grid_search_maze.py
```

##### atari 环境
```
cd cee
python pureppo/atari.py
```

注：对于cee方法，阈值调整在common/classify_points.py中的94行  #c_threshold = 0.8#

cee-woc方法，调整阈值在common/classify_points.py中129行，此处的阈值不能大，要偏小