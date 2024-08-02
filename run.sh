sudo apt-get remove --purge sqlite3

sudo apt-get update
sudo apt-get install build-essential
wget https://www.sqlite.org/2023/sqlite-autoconf-3390400.tar.gz
tar xvfz sqlite-autoconf-3390400.tar.gz
cd sqlite-autoconf-3390400
./configure
make
sudo make install


streamlit run app.py
