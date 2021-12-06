# CTVI-master
For  ICDM 2021. Accepted as short paper.

The implementation of the CTVI model(**C**itywide **T**raffic **V**olume **I**nference).

Paper: ```Temporal Multi-view Graph Convolutional Networks for Citywide Traffic Volume Inference```

# Usage:
## Install dependencies

```pip install -r requirements.txt```

## Clone this repo
```
git clone https://github.com/dsj96/CTVI-master.git
```

## Function
```args.py``` defines some necessary parameters.

```attention.py``` defines the multi-head temporal attention and position encoder model.

```extract_city_volume_info.py``` is adpoted  to extract and process raw city traffic volume data. 

```FNN.py``` is the implemention of three layers ```MLP```.

```jinan_optuna.py``` is the implemention of our model on Jinan dataset. You can run and evaluate the model by executing this code file.  And if you want to change the range of optuna hyperparameters, you can modify the ```objective``` function in ```jinan_optuna.py``` file. 

```metrics.py``` is used to evaluate our model and print log information.

```utils.py``` file is mainly used to implement some data processing functions.

```walk.py``` file is mainly used to generate random walk sequence on affinity graph.

```jinan.zip``` file is mainly to provide a toyset to run.

# Data
We conduct our experiments on Hangzhou and Jinan cities in China. Due to ```privacy issues```, we public part of the Jinan traffic vloume data in an anonymous form.
The preprocessed data is included in jinan.zip file (just unzip the file under the current file path). 

## Split Data
We randomly split the road segments with traffic volume data into training ```(80%)``` and testing ```(20%)```, respectively. We further select 20% of the training randomly as
validation. 


Note that for the selected road segments used for testing, we completely masked its traffic volume information in all time slice. Afterwards we use ```CTVI``` model to inference the traffic volume values in each time slice.

### Roadnet Data Format
```roadnet.txt```: intersection0_intersection1, num_of_lanes, speed limit, road segment name

```cams_attr.txt```: sensor ID, intersection0_intersection1, num_of_lanes, road grade, speed limit, road segment name
### Volume File Format
Each file contains the real time traffic volume information in the following form:

File name ```8_1_8_0_4.volume``` denotes the traffic volume values at August 1, 8:00:00 to 8:04:59.

In each file:
```5_6,172```: intersection0_intersection1, traffic volume


# Training and Evaluate
You can train and evaluate the model by run ```jinan_optuna.py``` file.

# Cite
Please cite our paper if you find this code useful for your research:
```
update later...
```

