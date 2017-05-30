cd src
python train.py new 15 32 16_32 32_64 96 sigmoid 0.75 2100000 None false
python examine.py model_15_32_16_32_32_64_96_sigmoid_0.75_210000_None_false new 
# test epoch
python train.py exist 25 32 16_32 32_64 96 sigmoid 0.75 210000 None false
python examine.py model_25_32_16_32_32_64_96_sigmoid_0.75_210000_None_false exist 
# test epoch 
python train.py exist 50 32 16_32 32_64 96 sigmoid 0.75 210000 None false
python examine.py model_50_32_16_32_32_64_96_sigmoid_0.75_210000_None_false exist 
# test drop out rate
python train.py exist 15 32 16_32 32_64 96 sigmoid 0.5 210000 None false
python examine.py model_15_32_16_32_32_64_96_sigmoid_0.5_210000_None_false exist 
# test for batch normalization, with low drop out rate
python train.py exist 15 32 16_32 32_64 96 sigmoid 0.98 210000 None true
python examine.py model_15_32_16_32_32_64_96_sigmoid_0.98_210000_None_true exist 
# test for l1 regularization 
python train.py exist 15 32 16_32 32_64 96 sigmoid 0.75 210000 l1_0.01 false
python examine.py model_15_32_16_32_32_64_96_sigmoid_0.75_210000_l1_0.01_false exist 
# test for l2 regularization
python train.py exist 15 32 16_32 32_64 96 sigmoid 0.75 210000 l2_0.01 true
python examine.py model_15_32_16_32_32_64_96_sigmoid_0.75_210000_l2_0.01_true exist 
# test for deep layer with relu
python train.py exist 20 32 16_32 32_64 96_96_96 relu 0.75 210000 None false
python examine.py model_20_32_16_32_32_64_96_96_96_relu_0.75_210000_None_false exist 
# test for deep layer with relu with l1 
python train.py exist 20 32 16_32 32_64 96_96_96 relu 0.75 210000 l1_0.001 false
python examine.py model_20_32_16_32_32_64_96_96_96_relu_0.75_210000_l1_0.001_false exist 
# test for deep layer with relu with l1 and normalization
python train.py exist 20 32 16_32 32_64 96_96_96 relu 0.75 210000 l1_0.001 true
python examine.py model_20_32_16_32_32_64_96_96_96_relu_0.75_210000_l1_0.001_true exist 
# test for question and answer dim
python train.py exist 15 32 20_20 20_20 40 sigmoid 0.75 210000 None false
python examine.py model_15_32_20_20_20_20_40_sigmoid_0.75_210000_None_false exist 
 







