# !/bin/bash

# train ssl encoder
python run_model.py --shuffle --scheduler --epochs 300 --batch_size 1024 --save_interval 5  --ssl --train

# train linear probe (SSL -> SL)
# map embeddings to 10 classes (without an act layer)
python run_model.py --shuffle --scheduler --epochs 15 --batch_size 128 --save_interval 5  --weight weights_ssl/model-100.pt --ssl --linear_probe

# test (with linear probe)
python run_model.py --batch_size 128 --weight weights_ssl_linear_probe/model-15.pt --ssl --test

# train supervised learning
python run_model.py --shuffle --scheduler --epochs 50 --batch_size 64 --save_interval 5  --sup --train

# test
python run_model.py --batch_size 64 --weight weights_supervised/model-50.pt --sup --test
