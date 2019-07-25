#!/bin/bash
if [ "$1" == "embedding" ]; then
  fileid="1BeBsSp_UMYT0KBPKlo5nl9duR3OZjcsy"
  filename="BERT-word-embedding.npy"
elif [ "$1" == "bert-param" ]; then
  fileid="1vtjpj3FM0HjlnoAvYrsYXB79sN7RLlnI"
  filename="bert_param.tar.gz"
fi


curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
