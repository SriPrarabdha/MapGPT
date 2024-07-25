DATA_ROOT=../datasets
outdir=${DATA_ROOT}/exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root /path/to/images
      --split MapGPT_72_scenes_processed
      --end 10  # the number of cases to be tested
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o-2024-05-13
      --response_format json
      --max_tokens 1000
      "

python vln/main_gpt.py $flag


python3 vln/main_gpt.py --root_dir datasets/ --img_root /media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/MapGPT/RGB_Observations --split MapGPT_72_scenes_processed --end 200  --output_dir ../datasets/exprs_map/test/ --max_action_len 15 --save_pred --stop_after 3  --response_format json --max_tokens 1000
python3 vln/main_gpt_new.py --root_dir datasets/ --img_root /media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/MapGPT/RGB_Observations --split MapGPT_72_scenes_processed --end 200  --output_dir ../datasets/exprs_map/test/ --max_action_len 15 --save_pred --stop_after 3  --response_format json --max_tokens 1000
