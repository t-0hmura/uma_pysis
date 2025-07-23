cd ./small/
uma_pysis input.yaml | tee pysis.log
python3 example.py

cd ..
cd ./large/
uma_pysis input.yaml | tee pysis.log