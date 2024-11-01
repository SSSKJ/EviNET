## Table 2(a)
python trian_edl.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table2a

## Table 2(b)
python all_in_one_block.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table2b --fix

## Table 2(c)
python all_in_one_block.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table2c

## Table 2(d)
python all_in_one_block_gcn.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table2d --fix

## Table 3 w/o AT
python all_in_one_gcn.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table3