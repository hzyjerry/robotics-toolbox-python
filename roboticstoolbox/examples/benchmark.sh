# cd Projects/Motion/robotics-toolbox-python/
# source activate pytorch3d


# python roboticstoolbox/examples/chomp.py --pts 1 --vb --nq 10 > data/verbose_pts0001_nq10.txt
# python roboticstoolbox/examples/chomp.py --pts 5 --vb --nq 10 > data/verbose_pts0005_nq10.txt
# python roboticstoolbox/examples/chomp.py --pts 5 --vb --nq 10 > data/verbose_pts0005_nq10.txt
# python roboticstoolbox/examples/chomp.py --pts 5 --vb --nq 10 > data/verbose_pts0005_nq10.txt
# python roboticstoolbox/examples/chomp.py --pts 5 --vb --nq 10 > data/verbose_pts0005_nq10.txt

# for i in 1 5 20 50 100 200 400 800 1600 3200 6400
# do
# 	a=`printf "%.4d" $i`
# 	echo "data/verbose_pts${a}_nq10.txt"
# 	python roboticstoolbox/examples/chomp.py --pts $i --vb --nq 10 > "data/verbose_pts${a}_nq10.txt"
# done


# for i in 1 5 20 50 100 200 400 800 1600 3200 6400
for i in 6400
do
	a=`printf "%.4d" $i`
	echo "data/noverb_pts${a}_nq10.txt"
	python roboticstoolbox/examples/chomp.py --pts $i --nq 10 > "data/noverb_pts${a}_nq10.txt"
done