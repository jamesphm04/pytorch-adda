"""Params for ADDA."""

# params for dataset and data loader
# data_root = "data"
# dataset_mean_value = 0.5
# dataset_std_value = 0.5
# dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
# dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 1
image_size = 1024

# params for source dataset
src_dataset = "source"
# src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
# src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
# src_model_trained = True

# params for target dataset
tgt_dataset = "target"
# tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
# tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
# d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
# num_epochs_pre = 10
# log_step_pre = 20
# eval_step_pre = 20
# save_step_pre = 100
num_epochs = 50
log_step = 10
save_step = 10
manual_seed = 44

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
