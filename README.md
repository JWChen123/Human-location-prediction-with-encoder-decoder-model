# ReadMe
## Human location prediction based on encoder-decoder model

This is a pytorch implementation of our prediction model. The method of processing dataset is borrowed from DeepMove:https://github.com/vonfeng/DeepMove.git.

# Artitecture
This project contains three parts including: codes,data,results
## Requirements:
* python 3.6.6
* torch 0.4.1
* numpy
* matplotlib

## Codes
main.py and main_batch.py are non-batch(batch=1) and batch version of project; train.py contains the method of processing training dataset; model.py contains the prediction model; sparse_traces.py is the method of preprocessing.

## Data
three preprocessed datasets
* foursquare.pk (https://github.com/vonfeng/DeepMove.git.)
* foursquare_2012.pk (original dataset[1])
* gowalla.pk (original dataset[2]).

## Results
It contains results and two pretrain models of two datasets as foursquare and foursquare_2012.
The result of gowalla will come soon.

Note that two pretrain models are non-batch version
# Implement

You can directly run Python main.py.
You can also directly run Python main_batch.py, which is the batch_version of our project.

## param 

the paramameters are listed in the results.

# Reference 
[1]:Yang, D.; Zhang, D.; Zheng, V. W.; Yu, Z. (2014): Modeling user activity preference by leveraging user spatial temporal characteristics in lbsns. IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 45, no. 1, pp. 129–142.

[2]:Cho, E.; Myers, S. A.; Leskovec, J. (2011): Friendship and mobility: user movement in location-based social networks,” Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, pp. 1082–1090.