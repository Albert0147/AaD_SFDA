## source pretrain

python image_source.py --trte val --da oda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 50 --s 0

python image_source.py --trte val --da oda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 50 --s 1

python image_source.py --trte val --da oda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 50 --s 2

python image_source.py --trte val --da oda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 50 --s 3


## adaptation

python tar_open.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output target_b1/ --beta 1

python tar_open.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 1 --output_src ckps/source/ --output target_b1/ --beta 1

python tar_open.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 2 --output_src ckps/source/ --output target_b1/ --beta 1

python tar_open.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 3 --output_src ckps/source/ --output target_b1/ --beta 1