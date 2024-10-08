# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os


import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding

import os
import sys
# print(os.getcwd())
path=os.path.abspath(os.path.dirname(__file__)+"/..")# 打印当前工作目录
sys.path.append(path)
# print(sys.path)  # 打印修改后的路径列表

from rewardbench import DPO_MODEL_CONFIG, DPOInference, load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section


# get token from HF_TOKEN env variable, but if it doesn't exist pass none
# HF_TOKEN = os.getenv("HF_TOKEN", None)
# # this is necessary to automatically log in when running this script in docker/batch beaker jobs
# if HF_TOKEN is not None:
#     from huggingface_hub._login import _login
#
#     _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument("--loss_data_path", type=str, default="ultrafeedback_tied/test/non_tie_data_test.jsonl", help="path of loss data")
    parser.add_argument("--evaluation_results",type=str, default=None, help="path of evaluation results")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--evaluation_mode",type=int,default=0,help="0 for original reward, 1 for dpo theta reward 2 for train and valid acc")
    parser.add_argument("--dpo_theta", type=float, default=-0.5, help="-alpha value in ToDO, please keep the same as training process")
    parser.add_argument("--dpo_beta", type=float, default=0.01, help="beta value in DPO/ToDO, please keep the same as training process")
    parser.add_argument("--train_and_valid_acc_path", type=str, default=None, help="save path")
    parser.add_argument("--debug", type=bool, default=False, help="use only 10 examples")
    parser.add_argument("--save_path_prefix",type=str, default="Results/reward_bench_results", help="save path prefix")
    parser.add_argument(
        "--model_abbr", type=str,default="reward_bench", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )

    args = parser.parse_args()
    return args

def evalution_acc_original_reward(args):
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")
    print(f"preference: {args.pref_sets}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("loading default dataset")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
    )

    dataset = dataset.remove_columns("id")
    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(30))
        subsets = subsets[:30]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }
    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model=model,
        ref_model=ref_model,
        beta=args.dpo_beta,
        theta=args.dpo_theta,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.build_tie_batch, remove_columns=column_names)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                          tie_inference=False)

        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()
        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(scores_chosen_batch, scores_rejected_batch)
        ]
        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["ref_model"] = args.ref_model
    results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    if ref_free:
        results_grouped["model_type"] = "DPO Ref. Free"
        save_modifier = "_ref_free"
    else:
        save_modifier = ""
    results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer,
                                                                         "chat_template") else "tokenizer"
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    final_res = {}
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
        final_res[subset] = num_correct / num_total
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)
    else:
        mean_score = np.mean(list(final_res.values()))
        print("final_res is: ", final_res, " means Prior is :", mean_score)
    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        args.model + save_modifier,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        save_path=f"{args.save_path_prefix}/{args.model_abbr}_results"
    )
    # if not args.do_not_save:
    #     logger.info(f"Uploaded reward model results to {results_url}")

    # upload chosen-rejected with scores
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = args.model
    scores_dict["model_type"] = "DPO"
    scores_dict["chat_template"] = args.chat_template
    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    scores_url = save_to_hub(
        scores_dict, args.model + save_modifier, sub_path_scores, args.debug, local_only=True,
        save_path=f"{args.save_path_prefix}/{args.model_abbr}_chosen_rejected_scores"
    )
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")
    if args.pref_sets:
        print("Prior results is ", np.mean(list(final_res.values())))


    # 加载数据集
def evaluation_reconstruct_reward(args,tie_type="tie_loss"):
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)
    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")
    print(f"preference: {args.pref_sets}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("loading default dataset")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
    )

    dataset = dataset.remove_columns("id")
    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(30))
        subsets = subsets[:30]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }
    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model=model,
        ref_model=ref_model,
        beta=args.dpo_beta,
        theta=args.dpo_theta,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.build_tie_batch, remove_columns=column_names)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    chosen_max_results=[]
    scores_chosen = []
    scores_tie = []
    scores_rejected = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                          tie_inference=True,tie_type=tie_type)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_tie_batch = [result["tie"] for result in reward_tie]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_tie_batch = reward_tie.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()
        for chosen, tie, rejected in zip(scores_chosen_batch, scores_tie_batch, scores_rejected_batch):
            if chosen==max(chosen,tie,rejected):
                chosen_max_results.append(1)
            else:
                chosen_max_results.append(0)
        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch
        scores_tie += scores_tie_batch
    out_dataset = dataset.add_column("chosen_max_results", chosen_max_results)
    # out_dataset = dataset.add_column("results", results)
    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)
    out_dataset = out_dataset.add_column("scores_tie", scores_tie)

    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["ref_model"] = args.ref_model
    results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    if ref_free:
        results_grouped["model_type"] = "DPO Ref. Free"
        save_modifier = "_ref_free"
    else:
        save_modifier = ""
    results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer,
                                                                         "chat_template") else "tokenizer"
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    final_res = {}
    for result_name in ["chosen_max_results"]:
        print(f"begin eval ===========  {result_name}  =================")

        for subset in present_subsets:
            subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
            num_correct = sum(subset_dataset[result_name])
            num_total = len(subset_dataset[result_name])
            final_res[subset] = num_correct / num_total
            print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
            results_grouped[subset] = num_correct / num_total
        if not args.pref_sets:
            results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
            print(results_leaderboard)
        print(f"end eval ===========  {result_name}  =================")

    # for subset in present_subsets:
    #     subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
    #     num_correct = sum(subset_dataset["results"])
    #     num_total = len(subset_dataset["results"])
    #     print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
    #     results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results


    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        args.model + save_modifier,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        save_path=f"{args.save_path_prefix}/{args.model_abbr}_results"
    )
    # if not args.do_not_save:
    #     logger.info(f"Uploaded reward model results to {results_url}")

    # upload chosen-rejected with scores
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = args.model
    scores_dict["model_type"] = "DPO"
    scores_dict["chat_template"] = args.chat_template
    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"
    scores_url = save_to_hub(
        scores_dict, args.model + save_modifier, sub_path_scores, args.debug, local_only=True,
        save_path=f"{args.save_path_prefix}/{args.model_abbr}_chosen_rejected_scores"
    )
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")
    if args.pref_sets:
        mean_score = np.mean(list(final_res.values()))
        print("final_res is: ", final_res, " means Prior is :", mean_score)

            ##compute mean score





def evaluation_ceartain_acc_of_train_and_valid(args,tie_type):
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")
    print(f"preference: {args.pref_sets}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"loading {args.loss_data_path}")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        training_data=args.loss_data_path,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "prompt", "tie"],
    )

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(30))
        subsets = subsets[:30]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }
    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model=model,
        ref_model=ref_model,
        beta=args.dpo_beta,
        theta=args.dpo_theta,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.build_tie_batch, remove_columns=column_names)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_tie = []
    scores_rejected = []
    tie_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                          tie_inference=True,tie_type=tie_type)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_tie_batch = [result["tie"] for result in reward_tie]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_tie_batch = reward_tie.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()
        for item, chosen, tie, rejected in zip(batch["tie"], scores_chosen_batch, scores_tie_batch,
                                               scores_rejected_batch):
            if item == True:
                if tie == max(chosen, tie, rejected):
                    results.append(1)
                else:
                    results.append(0)
            else:
                if chosen == max(chosen, tie, rejected):
                    results.append(1)
                else:
                    results.append(0)

        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch
        scores_tie += scores_tie_batch
        tie_labels += batch["tie"]

    print("evaluation acc")
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    # out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)
    out_dataset = out_dataset.add_column("scores_tie", scores_tie)
    assert args.train_and_valid_acc_path is not None
    if not os.path.exists(args.train_and_valid_acc_path):
        os.makedirs(args.train_and_valid_acc_path)
    import json
    with open(os.path.join(args.train_and_valid_acc_path, "score_results.json"), 'w') as f:
        f.write(json.dumps(
            {"scores_chosen": scores_chosen, "scores_rejected": scores_rejected, "scores_tie": scores_tie,
             "tie_labels": tie_labels, "results": results, "accuracy": sum(results) / len(results)}))
    print("validation accuracy is : ", sum(results) / len(results))


def evaluation_ceartain_acc_of_train_and_valid_orrginal_reward(args):
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")
    print(f"preference: {args.pref_sets}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"loading {args.loss_data_path}")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        training_data=args.loss_data_path,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "prompt", "tie"],
    )


    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(30))
        subsets = subsets[:30]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }
    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model=model,
        ref_model=ref_model,
        beta=args.dpo_beta,
        theta=args.dpo_theta,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.build_tie_batch, remove_columns=column_names)

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_tie = []
    scores_rejected = []
    tie_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                          tie_inference=False)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            # scores_tie_batch = [result["tie"] for result in reward_tie]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            # scores_tie_batch = reward_tie.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()
        for item, chosen, rejected in zip(batch["tie"], scores_chosen_batch,
                                               scores_rejected_batch):
            if item == True:
                continue
            else:
                if chosen == max(chosen,rejected):
                    results.append(1)
                else:
                    results.append(0)

        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch
        tie_labels += batch["tie"]

    print("evaluation acc")
    # out_dataset = dataset.add_column("results", results)
    #
    # # add subsets back (removed so it's not handled by cuda)
    # # out_dataset = out_dataset.add_column("subset", subsets)
    # # add scores_chosen and scores_rejected to the dataset
    # out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    # out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)
    # out_dataset = out_dataset.add_column("scores_tie", scores_tie)
    assert args.train_and_valid_acc_path is not None
    if not os.path.exists(args.train_and_valid_acc_path):
        os.makedirs(args.train_and_valid_acc_path)
    import json
    print("validation accuracy is : ", sum(results) / len(results))
    print("save data path is  : ",args.train_and_valid_acc_path+"/score_results.json")
    with open(os.path.join(args.train_and_valid_acc_path, "score_results.json"), 'w') as f:
        f.write(json.dumps(
            {"scores_chosen": scores_chosen, "scores_rejected": scores_rejected, "scores_tie": scores_tie,
             "tie_labels": tie_labels, "results": results, "accuracy": sum(results) / len(results)}))
    # print("validation accuracy is : ", sum(results) / len(results))


def evaluation_acc(args):
    import json
    accelerator = Accelerator()
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "prompt"],
    )

    load_out_dataset = json.load(open(os.path.join(args.evaluation_results, "scores.json")))
    results=[]
    for chosen,tie,rejected in zip(load_out_dataset["scores_chosen"],load_out_dataset["scores_tie"],load_out_dataset["scores_rejected"]):
        if chosen==max(chosen,tie,rejected):
            results.append(1)
        else:
            results.append(0)
        # if tie==min(chosen,tie,rejected):
        #     results.append(1)
        # else:
        #     results.append(0)
        if chosen>rejected:
            results.append(1)
        else:
            results.append(0)
    load_out_dataset.update({"results":results})
    out_dataset=dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", load_out_dataset["scores_chosen"])
    out_dataset = out_dataset.add_column("scores_tie", load_out_dataset["scores_tie"])
    out_dataset = out_dataset.add_column("scores_rejected",  load_out_dataset["scores_rejected"])
    # if args.tie_inference:
    #     out_dataset = out_dataset.add_column("scores_tie", scores_tie)
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")
    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["ref_model"] = args.ref_model
    results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    if ref_free:
        results_grouped["model_type"] = "DPO Ref. Free"
        save_modifier = "_ref_free"
    else:
        save_modifier = ""
    results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer,
                                                                         "chat_template") else "tokenizer"
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
        results_grouped[subset] = num_correct / num_total
    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

if __name__ == "__main__":
    args = get_args()
    if args.evaluation_results is not None:
        evaluation_acc(args)
    else:
        if args.evaluation_mode==0:
            print("evaluation acc of original reward")
            evalution_acc_original_reward(args)
        elif args.evaluation_mode==1:
            print("evaluation acc of tie reward")
            evaluation_reconstruct_reward(args,tie_type="tie_loss")
        else:
            raise ValueError("Unknown evaluation mode")

