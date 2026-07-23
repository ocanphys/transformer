## dev notes

- [x] checkpoints save both model and optimizer state. Optimizer state for adam tends to be larger than the model - consider keeping only the latest optimizer state but keep model weights for each checkpoint - this could be optional such that whenever we create a new checkpoint we can delete the  optimizer state from the previous checkpoint.


## infra notes
- device parameter control - hardcode vs config?
- 


## to do

- [x] write summary json at the end of each run.
- [x] rename logs.jsonl to train.jsonl also add JS/timezone friendly best for UI applications timestamp to each step
- think about queriable structure , sqlite db to save 
- have a config file that specifies runs path, where data is, where to cache tokenizers if we
- think about how to organize torch models - in this case we have transformer.py that is a model
- maybe there is way to keep track of diffs - ablations for architecture studies.
- git commit hash at the time of diff. 
- torch.compile - kernel fusion/hardware specific optimization
- torch.serve handles endpoints API
- build which can be run as a process: train.py --config configs/x.json for nohup/tmux/queue on N cluster nodes.


## visualization - organization options
- think about how to plot weights -
- for research/exploratory activities probably best to use jupyter notebook loading models at certain slices 
- this could be weight histograms, computational graphs, SAEs, anything thats mech interp sota.
- also think about visualizing the loss landscape during training. How model parameters are evolving collectively and 

##