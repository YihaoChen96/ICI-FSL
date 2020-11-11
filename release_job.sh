srun --gres gpu:4 -c 24 -p short -J "ici_train" --pty bash scripts/run_tiered_train.sh \
--account overcap \
-x asimo,jill,hal,ash,calculon,c3po,breq,johnny5,bmo,neo,rosie