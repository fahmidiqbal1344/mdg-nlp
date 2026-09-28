# !/bin/bash

# Compare the results of the two sigtest files


# time_datas        = ['event', 'time']
# dimensions        = ['base', 'length', 'adversarial', 'domain']

# Loop through the time_datas and dimensions
for task in time event; do
    for dimension in base length adversarial domain; do
        echo "Comparing $task and $dimension"
        # Compare the two files
        python3 base_results.py --step sigtest --task $task --dimension $dimension
    done
done