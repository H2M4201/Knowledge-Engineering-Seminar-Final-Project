conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
