#/bin/bash

wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
unzip gzip.zip
mv gzip/emnist-byclass-train-images-idx3-ubyte.gz train-images-idx3-ubyte.gz
mv gzip/emnist-byclass-train-labels-idx1-ubyte.gz train-labels-idx1-ubyte.gz
mv gzip/emnist-byclass-test-images-idx3-ubyte.gz t10k-images-idx3-ubyte.gz
mv gzip/emnist-byclass-test-labels-idx1-ubyte.gz t10k-labels-idx1-ubyte.gz

rm -rf gzip gzip.zip
