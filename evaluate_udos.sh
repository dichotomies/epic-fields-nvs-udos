
export VIDS=$(python -c "import os; print(' '.join([x.split('-')[0] for x in os.listdir('cache/task2')]))")

for VID in $VIDS; do python evaluate_udos.py --dir_cache=cache/task2 \
	--dir_output='outputs/task2/' \
	--model=mg --vid=$VID;
done

for VID in $VIDS; do python evaluate_udos.py --dir_cache=cache/task2 \
	--dir_output='outputs/task2/' \
	--model=neuraldiff --vid=$VID;
done
