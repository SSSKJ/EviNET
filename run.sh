## Amazon-photo
python all_in_one_block_gcn.py --dataset amazon-photo --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 55 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table1

## Amazon-computer
python all_in_one_block_gcn.py --dataset amazon-computer --u_hidden 64 --b2e_dropout 0.4 --b2e_layers 3 --b2e_lr 0.001 --hidden_channels 64 --gamma 15 --lr 0.0005 --dropout 0.2 --use_bn --runs 5 --prefix Table1

## Coauthor-cs
python all_in_one_block_gcn.py --dataset coauthor-cs --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.01 --hidden_channels 64 --gamma 55 --lr 0.001 --dropout 0.2 --use_bn --runs 5 --prefix Table1 

## Coauthor-physics
python all_in_one_block_gcn.py --dataset coauthor-physics --u_hidden 64 --b2e_dropout 0.6 --b2e_layers 3 --b2e_lr 0.01 --hidden_channels 64 --gamma 15 --lr 0.01 --dropout 0.2 --use_bn --runs 5 --prefix Table1 