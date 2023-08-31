
python evaluate_nvs.py --dir_cache=cache/task1 \
	--dir_output=outputs/task1 \
	--model=neuraldiff --action_type=within_action

python evaluate_nvs.py --dir_cache=cache/task1 \
	--dir_output=outputs/task1 \
	--model=neuraldiff --action_type=outside_action

python evaluate_nvs.py --dir_cache=cache/task1 \
	--dir_output=outputs/task1 \
	--model=neuraldiff --action_type=outside_action_easy
