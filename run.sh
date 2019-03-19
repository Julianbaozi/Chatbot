 echo "'">  log/att.txt
nohup python -u train_script.py --attn True   > log/att.txt &

echo "'"> log/nonatt.txt
nohup python -u train_script.py --attn False > log/nonatt.txt &