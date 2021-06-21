import codecs
import os
import pickle
import numpy as np 
import tqdm
from collections import Counter
import os
import nltk
import time
from mpt.mpt import MerklePatriciaTrie 

from keras.models import load_model

from utils import get_char_to_int, get_int_to_char, read_text_to_predict
from edlibutils import align_output_to_input
import argparse

#funcion que verifica si existe o no una sequencia en un trie dado
def exists_trie(trie_vocab, word):
    try:
        bword = bytes(word, 'utf-8')
        val = trie_vocab.get(bword)
        #print('the word %s exists in the trie'%str(word))
        return True
    except Exception as e:
        #print('%s, Not accessible in the trie'%str(word))
        return False 

#funcion que selecciona la mejor opcion de varias correcciones
# opciones = {'opcion1':freq1, 'opcion2':freq2, ...}
def select_option(trie_vocab, opciones):
    existen = {}
    for word in opciones:
        if exists_trie(trie_vocab, word):
            item = {word:opciones[word]}
            existen.update(item)
    if len(existen)==0:
        return None 
    else:
        existen_ordenado = sorted(existen, key = existen.get, reverse=True)
        return existen_ordenado[0]

def lstm_synced_correct_ocr(model, charset, text):
    # load model
    conf = model.get_config()
    conf_result = conf['layers'][0].get('config').get('batch_input_shape')
    seq_length = conf_result[1]
    #print(seq_length)
    char_embedding = False
    if conf['layers'][0].get('class_name') == u'Embedding':
        char_embedding = True
    with codecs.open(charset, 'r') as f:
        charset = f.read()
    n_vocab = len(charset)
    char_to_int = get_char_to_int(charset)
    int_to_char = get_int_to_char(charset)
    lowercase = True
    for c in u'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if c in charset:
            lowercase = False
            break

    pad = u'\n'
    #print(charset)
    to_predict = read_text_to_predict(text, seq_length, lowercase,
                                      n_vocab, char_to_int, padding_char=pad,
                                      char_embedding=char_embedding)

    outputs = []
    inputs = []

    predicted = model.predict(to_predict, verbose=0)
    for i, sequence in enumerate(predicted):
        #for p in sequence:
            #print("aqui")
        predicted_indices = [np.random.choice(n_vocab, p=p) for p in sequence]
        pred_str = u''.join([int_to_char[j] for j in predicted_indices])
        outputs.append(pred_str)

        if char_embedding:
            indices = to_predict[i]
        else:
            indices = np.where(to_predict[i:i+1, :, :] == True)[2]
        inp = u''.join([int_to_char[j] for j in indices])
        inputs.append(inp)

    idx = 0
    counters = {}

    for input_str, output_str in zip(inputs, outputs):
        if pad in output_str:
            output_str2 = align_output_to_input(input_str, output_str)
        else:
            output_str2 = output_str
        for i, (inp, outp) in enumerate(zip(input_str, output_str2)):
            if not idx + i in counters.keys():
                counters[idx+i] = Counter()
            counters[idx+i][outp] += 1

        idx += 1

    agg_out = []
    for idx, c in counters.items():
        agg_out.append(c.most_common(1)[0][0])

    corrected_text = u''.join(agg_out)
    corrected_text = corrected_text.replace(pad, u'')
    #print(corrected_text)
    return corrected_text 

def corrigir(model, charset, word, iterar, trie_vocab):
    #resp = {word:1}
    resp = {}
    for i in range(iterar):
        try:
            word_corr = lstm_synced_correct_ocr(model, charset,word)
        except Exception as e:
            word_corr = word 
        try:
            resp[word_corr]+=1
        except Exception as e:
            resp.update({word_corr:1})
    return resp 

def corrigir_sent(model, charset, sent, iterar, trie_vocab):
    sent_mod = []
    words = nltk.word_tokenize(sent.strip())
    #print('words', words)
    for word in tqdm.tqdm(words):
        # if word in vocab, pasar si corregir
        if word not in '{}()[].,:;+-*/&|<>=~':
            aux = exists_trie(trie_vocab, word)
            if aux is True: 
                selected_word = word 
            else:
                print('corregir: %s'%word)
                selected_word = ''
                resp = corrigir(model, charset, word, iterar, trie_vocab)
                print(resp)
                selected_word = select_option(trie_vocab, resp)
                print('selectec: ',selected_word)
 
        else:
            selected_word = word 
        # concatenar
        if selected_word is None:
            selected_word = word 
        sent_mod.append(selected_word)
    return ' '.join(sent_mod)

def corrigir_line(model, charset, line, iterar, trie_vocab):
    sents = nltk.sent_tokenize(line)
    sents_new = []
    for sent in sents:
        sent_new = corrigir_sent(model, charset, sent, iterar, trie_vocab)
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

def processar_onefile(pathin, pathout, txtfile, model, charset, iterar, trie_vocab):
    lines = read_file(os.path.join(pathin, txtfile))
    out = codecs.open(os.path.join(pathout, txtfile), 'w', 'utf-8')
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line_new = corrigir_line(model, charset, line, iterar, trie_vocab)
        out.write(line_new+'\n')
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('scrip para correção com ochre')
    parser.add_argument('--folderin', 
                        type=str,
                        required=False,
                        help='caminho para a pasta de entrada')
    parser.add_argument('--folderout', 
                        required=False,
                        default='ochre_trie',
                        type=str,
                        help='caminho para a pasta de saida')
    parser.add_argument('--it', 
                        type=int,
                        required=False,
                        default=1,
                        help='número de vezes que corrige um mesmo texto (default 1)')
    

    # read args
    args = parser.parse_args()
    folderin=args.folderin 
    it = args.it
    folderout=args.folderout 
    
    # load model and trie
    print("carregar modelo")
    model_path = '../ochre_app/models/0.1241-88.hdf5'
    model = load_model(model_path)
    charset = '../ochre_app/models/chars-lower.txt'

    # vocab trie unigrams
    with open('../vocab_union.trie', 'rb') as f:
        trie_vocab = pickle.load(f)

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
            #obter arquivos já processados
            files_ok = os.listdir(folderout)
            #print("processar todos los archivos de una carpeta in folder: %s" %files)
            for fil in files:
                try:
                    if fil not in files_ok:
                        processar_onefile(folderin, folderout, fil, model, charset, it, trie_vocab)
                    else:
                        print("arquivo %s já processado"%fil)
                except Exception as e:
                    print('error in %s'%fil)
                    print(e)
        else:
            print("folder de entrada no existe")
