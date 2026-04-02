# !/bin/bash

# -- supervised learning (conventional model training)
# -- train
python run_model.py --shuffle --scheduler --epochs 50 --batch_size 64 --save_interval 5 --mode sup --train

# -- test (classification)
python run_model.py --batch_size 1024 --weight sup/weights/model-50.pt --mode sup --test


# -- ssl (cnn)
# -- train encoder
python run_model.py --shuffle --scheduler --epochs 300 --batch_size 1024 --save_interval 5 --mode ssl --train

# -- train linear probe (SSL -> SL)
python run_model.py --shuffle --scheduler --epochs 15 --batch_size 128 --save_interval 5 --weight ssl/weights/model-300.pt --mode ssl --linear_probe

# -- test (classification)
python run_model.py --batch_size 1024 --weight ssl_lp/weights/model-15.pt --mode ssl --test


# -- ssl (vit)
# -- train encoder
python run_model.py --shuffle --scheduler --epochs 300 --batch_size 1024 --save_interval 5 --mode ssl_vit --train

# -- train linear probe (SSL -> SL)
python run_model.py --shuffle --scheduler --epochs 15 --batch_size 128 --save_interval 5 --weight ssl_vit/weights/model-300.pt --mode ssl_vit --linear_probe

# -- test (classification)
python run_model.py --batch_size 1024 --weight ssl_vit_lp/weights/model-15.pt --mode ssl_vit --test


# -- visualization
python tsne_plot.py
python umap_plot.py