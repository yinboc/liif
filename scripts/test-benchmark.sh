echo 'set5' &&
echo 'x2' &&
python test.py --config ./configs/test/test-set5-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/test/test-set5-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/test/test-set5-4.yaml --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-set5-6.yaml --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-set5-8.yaml --model $1 --gpu $2 &&

echo 'set14' &&
echo 'x2' &&
python test.py --config ./configs/test/test-set14-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/test/test-set14-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/test/test-set14-4.yaml --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-set14-6.yaml --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-set14-8.yaml --model $1 --gpu $2 &&

echo 'b100' &&
echo 'x2' &&
python test.py --config ./configs/test/test-b100-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/test/test-b100-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/test/test-b100-4.yaml --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-b100-6.yaml --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-b100-8.yaml --model $1 --gpu $2 &&

echo 'urban100' &&
echo 'x2' &&
python test.py --config ./configs/test/test-urban100-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test.py --config ./configs/test/test-urban100-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test.py --config ./configs/test/test-urban100-4.yaml --model $1 --gpu $2 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-urban100-6.yaml --model $1 --gpu $2 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-urban100-8.yaml --model $1 --gpu $2 &&

true
