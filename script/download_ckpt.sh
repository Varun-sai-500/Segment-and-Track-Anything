mkdir ckpt

# download aot-ckpt
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o ./ckpt/sam_vit_b_01ec64.pth

curl -L "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1" \
  -o ./ckpt/audio_mdl.pth

curl -L https://raw.githubusercontent.com/YuanGongND/ast/master/egs/audioset/data/class_labels_indices.csv \
  -o ./ckpt/class_labels_indices.csv
