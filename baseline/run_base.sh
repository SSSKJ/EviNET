for data in "wikics" "amazon-photo" "coauthor-cs" "coauthor-physics" "amazon-computer"
do
    for noise in 0 0.0001 0.01 0.1 0.5 1
    do
        for T in 1 5 10 20 50 100
        do
            python main.py --method ODIN --dataset $data --device 1 --noise $noise --T $T --run 5 --use_bn --epoch 1000 --prefix ${noise}_${T}
        done
    done
done

for dataset in  "wikics" "amazon-photo" "coauthor-cs" "coauthor-physics" "amazon-computer"
do
    for method in "ODIN"
    do
        python main.py --method $method --dataset $dataset --device 1 --run 5 --use_bn --epoch 1000
    done
done

for dataset in  "wikics" "amazon-photo" "coauthor-cs" "coauthor-physics" "amazon-computer"
do
    for method in "doctor" "maxlogits" "CRL"
    do
        python main.py --method $method --dataset $dataset --device 1 --run 5 --use_bn --epoch 1000
    done
done

for dataset in  "wikics" "amazon-photo" "coauthor-cs" "coauthor-physics" "amazon-computer"
do
    for method in "GNNSafe"
    do
        python main.py --method $method --dataset $dataset --device 1 --run 5 --use_bn --epoch 1000 --use_prop
    done
done

for dataset in "wikics" "amazon-photo" "coauthor-cs" "coauthor-physics" "amazon-computer"
do
    for method in "SGCN" "GPN"
    do
        for GPN_detect_type in "Alea" "Epist"
        do
            python main.py --method $method --dataset $dataset --GPN_detect_type $GPN_detect_type --device 1 --run 5 --epoch 1000
        done
    done
done
