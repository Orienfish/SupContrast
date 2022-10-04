for f in save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/*
do
  echo $f;
  python main_knn.py --dataset cifar10 --batch_size 128 --ckpt $f
  python main_knn.py --dataset cifar100 --batch_size 128 --ckpt $f
done

for f in save/SupCon/cifar100_models/SimCLR_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.5_trial_0_cosine/*
do
  echo $f;
  python main_knn.py --dataset cifar10 --batch_size 128 --ckpt $f
  python main_knn.py --dataset cifar100 --batch_size 128 --ckpt $f
done

