#!/bin/bash

n_steps=("10 25 50 75 100")
partitions=("1 5 10 20 100")



for i in $n_steps
do
   echo "Starting n_steps of $i: "
   
   python src/main.py --n_steps $i --skip_real True > approx_auc_nsteps_"$i"_partition_1_labels_100to10000.txt &
   
   echo "Done with $i."
done

wait
echo "All n steps are done!"

for i in $partitions
do
   echo "Starting partition of $i: "
   
   python src/main.py --partitions $i --label_path data/labels_100000.txt --prediction_path data/pred_cons_100000.txt --skip_real True > approx_auc_nsteps_100_partition_"$i"_labels_100000.txt &
   
   echo "Done with $i."
done

wait
echo "All partitions are done!"

# real
python src/main.py --skip_approx True > real_auc_labels_100to10000.txt &
python src/main.py --label_path data/labels_100000.txt --prediction_path data/pred_cons_100000.txt --skip_approx True > real_auc_labels_100000.txt &

wait
echo "All jobs done!"
