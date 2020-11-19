# python main.py --resume ckpt/res12_tiered.pth.tar --dataset tieredimagenet --mode test --num-workers 8 --unlabel 80 --num_shot 1 --num_test_ways 5 --folder dataset --gpu "0,1,2,3"
# python main.py --resume ckpt/res12_tiered.pth.tar --dataset tieredimagenet --folder dataset --unlabel 15 --num_shot 1 --num_test_ways 5 --data_picker fixmatch
python main.py --resume ckpt/tieredimagenet/res12_best.pth.tar --dataset tieredimagenet --folder dataset --unlabel 15 --num_shot 1 --num_test_ways 5 --data_picker fixmatch_torch --fixmatch_threshold 0.7