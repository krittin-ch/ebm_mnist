#!/bin/bash

# -- test sup
python run_model.py --batch_size 1024 --weight sup/weights/model-50.pt --mode sup --test

# -- test ssl
python run_model.py --batch_size 1024 --weight ssl_lp/weights/model-15.pt --mode ssl --test

# -- test ssl vit
python run_model.py --batch_size 1024 --weight ssl_vit_lp/weights/model-15.pt --mode ssl_vit --test
