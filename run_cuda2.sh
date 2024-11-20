module load stack/2024-04  gcc/8.5.0 cuda/11.8.0
# conda env create -p ../../conda_envs/Grid4D -f environment.yaml

source ~/.bashrc

conda activate ../../conda_envs/gs

# conda env create -p ../../conda_envs/gs --file ./environment.yml 

python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_20/ --eval --white_background  --time 20 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_20.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_40/ --eval --white_background  --time 40 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_40.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_60/ --eval --white_background  --time 60 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_60.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_80/ --eval --white_background  --time 80 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_80.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_100/ --eval --white_background  --time 100 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_100.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_120/ --eval --white_background  --time 120 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_120.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_140/ --eval --white_background  --time 140 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_140.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_160/ --eval --white_background  --time 160 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_160.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_180/ --eval --white_background  --time 180 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_180.log
python train.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_200/ --eval --white_background  --time 200 --load_image_on_the_fly --port 7009 > ./output/scale_exp2_put_fruit_train_200.log


python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_20/ --eval --white_background  --time 20 --load_image_on_the_fly> ./output/scale_exp2_put_fruit_render_20.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_40/ --eval --white_background  --time 40 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_40.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_60/ --eval --white_background  --time 60 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_60.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_80/ --eval --white_background  --time 80 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_80.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_100/ --eval --white_background  --time 100 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_100.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_120/ --eval --white_background  --time 120 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_120.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_140/ --eval --white_background  --time 140 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_140.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_160/ --eval --white_background  --time 160 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_160.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_180/ --eval --white_background  --time 180 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_180.log
python render.py -s ../4DGaussians/data/put_fruit/ -m ./output/scale_exp2/put_fruit_200/ --eval --white_background  --time 200 --load_image_on_the_fly > ./output/scale_exp2_put_fruit_render_200.log
