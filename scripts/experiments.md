Experiment Scripts 


Local Views: 2
dazzling-haze-8
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V=4 \
  +proj_dim=2048 \
  +lr=5e-4\
  +bs=256 \
  +epochs=100 \ 
  +num_workers=12 \
  +device="cuda"  \
  +prefetch_factor=4 \

Local Views: 4
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V=6 \
  +proj_dim=2048 \
  +lr=5e-4\
  +bs=256 \
  +epochs=100 \ 
  +num_workers=12 \
  +device="cuda"  \
  +prefetch_factor=4 \

Local Views: 8
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V=10 \
  +proj_dim=2048 \
  +lr=5e-4\
  +bs=256 \
  +epochs=100 \ 
  +num_workers=12 \
  +device="cuda"  \
  +prefetch_factor=4 \

Local Views: 12
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V=14 \
  +proj_dim=2048 \
  +lr=5e-4\
  +bs=256 \
  +epochs=100 \ 
  +num_workers=12 \
  +device="cuda"  \
  +prefetch_factor=4 \

Global, no Local 
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=8 \
  +V_local=0 \
  +model_name=vit_base_patch16_224.dino \
  +save_prefix=vit_base \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=128 \
  +lr=5e-4 \
  +bs=256 \
  +epochs=100 \
  +num_workers=4 \
  +device=cuda \
  +prefetch_factor=4 \
  +view_selection=random

# large
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +model_name=vit_large_patch14_reg4_dinov2.lvd142m \
  +save_prefix=vit_large \
  +global_img_size=224 \
  +local_img_size=98 \
  +proj_dim=2048 \
  +lr=5e-4 \
  +bs=256 \
  +epochs=100 \
  +num_workers=4 \
  +device=cuda \
  +prefetch_factor=4 \
  +view_selection=random

# Resnet
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +model_name=resnet50 \
  +save_prefix=resnet50 \
  +global_img_size=224 \
  +local_img_size=98 \
  +proj_dim=2048 \
  +lr=5e-4 \
  +bs=256 \
  +epochs=100 \
  +num_workers=4 \
  +device=cuda \
  +prefetch_factor=4 \
  +view_selection=random


# Mixed 8 Local Views 4/8 different Image, Same Class
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=4 \
  +V_mixed=4 \
  +model_name=vit_base_patch16_224.dino \
  +save_prefix=vit_cross_instance \
  +global_img_size=224 \
  +local_img_size=96 \
  +proj_dim=512 \
  +lr=5e-4 \
  +bs=256 \
  +epochs=100 \
  +num_workers=4 \
  +device=cuda \
  +prefetch_factor=4 \
  +view_selection=mixed \ 
  +benchmark=False \ 
  +dataset=cifar10