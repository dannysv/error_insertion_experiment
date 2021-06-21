import codecs
import os
import numpy as np 
import tqdm
import os
import nltk
import time
import sys, xmltodict
import os.path
import ktrain
import string, re
from word_correction.word_correction import WordCorrection

import argparse

def test_valid_word(word, index):
    if index != 0 and word[0].isupper() and word[1:].islower():
        return False, 1
    if word.isupper():
        return False, 2
    test = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', word)
    if len(test)==0:
        return False, 3
    return True, 0

def corrigir_sent(sentence, corrector):
    sent_mod = []
    for t, word in enumerate(nltk.word_tokenize(sentence)):
        #print(t, word)
        valid_word, test = test_valid_word(word, t)
        if valid_word == True:
            sent_mod.append(corrector.classify_input(word))
        else:
            sent_mod.append(word)
    final_sentence = ' '.join(sent_mod)
    return final_sentence 

def corrigir_line(line, corrector, predictor):
    sents = nltk.sent_tokenize(line)
    sents_new = []
    for sent in sents:
        if predictor.predict(sent) == 'corrige':
            sent_new = corrigir_sent(sent, corrector)
        else:
            sent_new = sent 
        sents_new.append(sent_new)
        pass
    return ' '.join(sents_new)

def read_file(path):
    try:
        with codecs.open(path, 'r', encoding="utf-8") as f:
            resp = f.readlines()
        print('read %s with utf-8'%path)
        return resp 
    except Exception as e:
        if "'utf-8' codec can't" in str(e):
        #try to read with iso-8859-1
            try: 
                with codecs.open(path, 'r', encoding="iso-8859-1") as f:
                    resp = f.readlines()
                print('read %s with iso-8859-1'%path)
                return resp 
            except Exception as e:
                print(e)
                return None
        else:
            print(e)
            return None

def processar_onefile(pathin, pathout, txtfile, corrector, predictor):
    lines = read_file(os.path.join(pathin, txtfile))
    out = codecs.open(os.path.join(pathout, txtfile), 'w', 'utf-8')
    for line in tqdm.tqdm(lines):
        line = line.strip()
        if len(line)>0:
            line_new = corrigir_line(line, corrector, predictor)
        else:
            line_new = line 
        out.write(line_new+'\n')
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('scrip para correcao com ochre')
    parser.add_argument('--folderin', 
                        type=str,
                        required=False,
                        help='caminho para a pasta de entrada')
    parser.add_argument('--folderout', 
                        required=False,
                        default='ochre_trie',
                        type=str,
                        help='caminho para a pasta de saida')
    

    # read args
    args = parser.parse_args()
    folderin=args.folderin 
    #it = args.it
    folderout=args.folderout 
    if os.path.exists(folderout):
        pass
    else:#create
        os.mkdir(folderout)
    
    # load resources to correct
    corrector = WordCorrection()
    predictor = ktrain.load_predictor("pred_bert")
   
    # process files of a given folder
    if folderin is not None:
        print('aqui')
        if os.path.exists(folderin):
            #listar os arquivos de txt
            files = os.listdir(folderin)
            print(files)
            files = [f for f in files if str(f).endswith('.txt') and 'readme' not in str(f)]

            if os.path.exists(folderout)==False:
                os.mkdir(folderout)
            #obter arquivos ja processados
            files_ok = os.listdir(folderout)
            #print("processar todos los archivos de una carpeta in folder: %s" %files)
            for fil in tqdm.tqdm(files):
                try:
                    if fil not in files_ok:
                        processar_onefile(folderin, folderout, fil, corrector, predictor)
                    else:
                        print("arquivo %s ja processado"%fil)
                except Exception as e:
                    print('error in %s'%fil)
                    print(e)
        else:
            print("folder de entrada no existe")
