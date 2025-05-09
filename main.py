"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder, ResnetEncoder
from utils import get_data_loader, init_model, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    # src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_encoder = init_model(net=ResnetEncoder(pretrained=True))
    # src_classifier = init_model(net=LeNetClassifier(),
    #                             restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=ResnetEncoder(pretrained=False))
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims))

    # # train source model
    # print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)

    # if not (src_encoder.restored and src_classifier.restored and
    #         params.src_model_trained):
    #     src_encoder, src_classifier = train_src(
    #         src_encoder, src_classifier, src_data_loader)

    # # eval source model
    # print("=== Evaluating classifier for source domain ===")
    # eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    # print(">>> Target Encoder <<<")
    # print(tgt_encoder)
    # print(">>> Critic <<<")
    # print(critic)

    # # init weights of target encoder with those of source encoder
    # if not tgt_encoder.restored:
    #     tgt_encoder.load_state_dict(src_encoder.state_dict())

    # if not (tgt_encoder.restored and critic.restored and
    #         params.tgt_model_trained):
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                            src_data_loader, tgt_data_loader)

    # # eval target encoder on test set of target dataset
    # print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    # eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    # print(">>> domain adaption <<<")
    # eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
