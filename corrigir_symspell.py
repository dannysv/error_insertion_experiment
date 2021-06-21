from symspellpy.symspellpy import SymSpell 
import codecs
import os
import numpy as np 
import tqdm
import os
import nltk
import time

import argparse

def corrigir_sent(sentence, sym_spell):
    sent_mod = []
    suggestions = sym_spell.lookup_compound(sentence, 
            max_edit_distance_lookup, transfer_casing=True, ignore_non_words=False) 

    if len(suggestions) != 0:
        final_sentence = suggestions[0].term
    else:
        final_sentence = sent
    return final_sentence 

def corrigir_line(line, sym_spell):
    sents = nltk.sent_tokenize(line)
    sents_new = []
    for sent in sents:
        sent_new = corrigir_sent(sent, sym_spell)
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

def processar_onefile(pathin, pathout, txtfile, sym_spell):
    lines = read_file(os.path.join(pathin, txtfile))
    out = codecs.open(os.path.join(pathout, txtfile), 'w', 'utf-8')
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line_new = corrigir_line(line, sym_spell)
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
    #parser.add_argument('--it', 
    #                    type=int,
    #                    required=False,
    #                    default=1,
    #                    help='número de vezes que corrige um mesmo texto (default 1)')
    

    # read args
    args = parser.parse_args()
    folderin=args.folderin 
    #it = args.it
    folderout=args.folderout 
    
    # load resources to correct
    #dict
    dic_path = "/hddcrucial/projects/learn/symspell/data/"
    max_edit_distance_dictionary = 2 
    prefix_length = 7 

    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = "1grams_freq.txt_inv" 
    bigram_path = "2grams_freq.txt_inv"
    if not sym_spell.load_dictionary(dic_path+dictionary_path, term_index=1, separator= "\t",
            count_index=0, encoding='utf-8'):
        print("Dictionary file not found")
    if not sym_spell.load_bigram_dictionary(dic_path+bigram_path, term_index=0,
            count_index=1, encoding='utf-8'): 
        print("Bigram dictionary file not found")  
    max_edit_distance_lookup = 2


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
                        processar_onefile(folderin, folderout, fil, sym_spell)
                    else:
                        print("arquivo %s já processado"%fil)
                except Exception as e:
                    print('error in %s'%fil)
                    print(e)
        else:
            print("folder de entrada no existe")
