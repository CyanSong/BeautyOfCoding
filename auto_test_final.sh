cd src
python train.py new 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false adam_0.001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false adam_0.001
python train.py exist 20 32 32_32 48_48 96 sigmoid 0.76 230000 None false adam_0.001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None true adam_001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false adam_0.0005
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false adam_0.0001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false rmsprop_0.001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false rmsprop_0.001
python train.py exist 20 32 32_32 48_48 96 sigmoid 0.76 230000 None false rmsprop_0.001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None true rmsprop_001
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false rmsprop_0.0005
python train.py exist 20 32 40_48 60_72 128 sigmoid 0.76 230000 None false rmsprop_0.0001
python train.py exist 20 32 50_50 100_10 150 sigmoid 0.76 230000 None false adam_0.0005