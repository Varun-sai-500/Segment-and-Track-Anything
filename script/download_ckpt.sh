mkdir ckpt

# download aot-ckpt
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o ./ckpt/sam_vit_b_01ec64.pth
