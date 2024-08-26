We direct modify the implementation of original Reward Bench to evaluate the preference modeling ability of DPO and ToDO 

## Quick Usage

To install for quick usage, install with pip as:
```shell
pip install -e .
```

For DPO evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_beta 0.01 \
--evaluation_mode 0 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results \ #save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8

CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_beta 0.01 \
--evaluation_mode 0 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results \ #save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8 \
--prior_datsets 
```

For ToDO evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py --model  ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_theta -0.5 \
--dpo_beta 0.01 \ 
--evaluation_mode 1 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results/ \save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8


CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py --model  ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_theta -0.5 \
--dpo_beta 0.01 \ 
--evaluation_mode 1 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results/ \save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8 \
--prior_datsets 
```

#### 


## Citation
```
@misc{lambert2024rewardbench,
      title={RewardBench: Evaluating Reward Models for Language Modeling}, 
      author={Nathan Lambert and Valentina Pyatkin and Jacob Morrison and LJ Miranda and Bill Yuchen Lin and Khyathi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2403.13787},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
