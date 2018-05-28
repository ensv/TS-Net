#CUHK
cd ./cuhk
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n0Fvr6CWmRSAGd2P7B-HX6W6hclJ_-Fw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n0Fvr6CWmRSAGd2P7B-HX6W6hclJ_-Fw" -O cuhk.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz cuhk.tar.gz

# vedai
cd ../vedai
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VA3V_Ox0jRMUYlINOblbIeU6_tF63Rdz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VA3V_Ox0jRMUYlINOblbIeU6_tF63Rdz" -O vedai.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz vedai.tar.gz

# nirscene
cd ../nirscene
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hC4LDIAIF8ZzVJ52YrTSOaNTtXXQQ-zF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hC4LDIAIF8ZzVJ52YrTSOaNTtXXQQ-zF" -O nirscene.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz nirscene.tar.gz