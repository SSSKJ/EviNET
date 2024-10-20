dataset=$1
device=$2
# echo "Running with dataset: $dataset and device: $device"

for gamma in 15 55 95 135
do
    for lr in 0.01 0.001 0.0005
    do
        for dropout in 0.2 0.4 0.6
        do
            for u_hidden in 64 128
            do
                for b2e_layers in 3 4
                    do
                    for b2e_dropout in 0.2 0.4 0.6
                    do
                        for b2e_lr in 0.01 0.001 0.0005
                        do
                            python all_in_one_block_gcn.py --dataset $dataset --device $device --u_hidden $u_hidden --b2e_dropout $b2e_dropout --b2e_layers $b2e_layers --b2e_lr $b2e_lr --hidden_channels 64 --gamma $gamma --aggr mean --lr $lr --dropout $dropout --use_bn --prefix ps --runs 5
                        done
                    done
                done
            done
        done
    done
done
