# !/bin/bash

# -- supervised learning (conventional model training)
# -- train
python run_model.py --shuffle --scheduler --epochs 50 --batch_size 64 --save_interval 5 --sup --train

# -- test (classification)
python run_model.py --batch_size 1024 --weight sup/weights/model-50.pt --sup --test


# -- ssl (encoder from supervised learning)
# -- train encoder
python run_model.py --shuffle --scheduler --epochs 300 --batch_size 1024 --save_interval 5 --ssl --train

# -- train linear probe (SSL -> SL)
python run_model.py --shuffle --scheduler --epochs 15 --batch_size 128 --save_interval 5 --weight ssl/weights/model-300.pt --ssl --linear_probe

# -- test (classification)
python run_model.py --batch_size 1024 --weight ssl_lp/weights/model-15.pt --ssl --test


# -- ssl (vit)
# -- train encoder
python run_model.py --shuffle --scheduler --epochs 300 --batch_size 1024 --save_interval 5 --ssl --vit --train

# -- train linear probe (SSL -> SL)
python run_model.py --shuffle --scheduler --epochs 15 --batch_size 128 --save_interval 5 --weight ssl_vit/weights/model-300.pt --ssl --vit --linear_probe

# -- test (classification)
python run_model.py --batch_size 1024 --weight ssl_vit_lp/weights/model-15.pt --ssl --vit --test


# -- visualization
python tsne_plot.py
python umap_plot.py