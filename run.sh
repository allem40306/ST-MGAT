PEMS08(){
    python train.py --in_dim=3 --out_dim=3 --num_nodes=170 --adj_path=./data/PEMS08/PEMS08.csv --data_path=./data/PEMS08 --save=./experiment/PEMS08_$1/
}

MRT(){
    python train.py --in_dim=123 --out_dim=119  --seq_len 12 --pre_len 12 --num_nodes=119 --adj_path=./data/MRT/MRT.csv --data_path=./data/MRT --save=./experiment/MRT_$1/
}

Youbike(){  
    python train.py --in_dim=511 --out_dim=2 --num_nodes=507 --batch_size=8 --adj_path=./data/$1/Youbike.csv --data_path=./data/$1 --save=./experiment/$1_$2/ --dataset $1
}

# PEMS08 r01
# MRT r01
# Youbike Youbike_1 r01