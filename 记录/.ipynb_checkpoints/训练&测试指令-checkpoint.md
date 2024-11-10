# 训练
python main.py --config ./config/ctrgcn_V1_J.yaml --device 0
# 测试
python main.py --config ./config/mixformer_V1_J.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_J.pt --device 0