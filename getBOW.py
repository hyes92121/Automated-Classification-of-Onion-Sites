from bs4 import BeautifulSoup
import requests
import os
import pandas
import time 
import subprocess
import sys


def getOnionText(url, proxies):
    print('Requesting onion from: {} .... '.format(url), end='')
    try:
        # url = 'http://hss33mlbykbsxmug.onion'
        html = requests.get(url, proxies=proxies).text
        soup = BeautifulSoup(html, "html5lib")
        # print(soup)
        # remove CSS and JS
        cleaned = (''.join(soup.findAll(text=lambda text: text.parent.name != "script" and text.parent.name != "style")))
        soup = BeautifulSoup(cleaned, "html5lib")
        # remove HTML comments
        for element in soup(text=lambda text: isinstance(text, Comment)):
            element.extract()
        text = soup.text
        print('Suceeded. Time elapsed: {}'.format(time.time()-s))
    except:
        print('Connection timed out. Passing onion...')
        text = ''

    return text

def crawl(csv_dir):
    d = os.listdir(csv_dir)
    for i, csv_file in enumerate(d):
        cat_dir = csv_file.split('.')[0]
        if not os.path.exists(cat_dir):
            os.mkdir(cat_dir)

        file_path = os.path.join(csv_dir, csv_file)
        df = pandas.read_csv(file_path, sep=',', header=None)

        for idx in range(len(df)):
            url = df[0][idx]
            title = df[1][idx]
            
            try:
                text = getOnionText(url)
            except KeyboardInterrupt:
                exit(-1)

            if text:
                with open('{}/{}000{}'.format(cat_dir, i, idx), 'w', encoding='utf-8') as f:
                    f.write('{} \n'.format(title))
                    f.write('{} \n'.format(text))

def get_wordgrp(file, WORD_GRP):
    # generate word groups
    if os.path.exists(WORD_GRP):
        process = subprocess.call(['rm', '-rm', WORD_GRP])
    else:
        os.makedirs(WORD_GRP)
    for cat in dirs:
        cat_path = 'data/'+cat
        for d in [f for f in os.listdir(cat_path) if '.DS' not in f]:
            process = subprocess.Popen(['java', 'GenerateWordGrp', '{}/{}'.format(cat_path, d)], stdout=subprocess.PIPE)
            stdout = process.communicate()[0]
            wrdgrp = stdout.decode('UTF-8').split('\n')
            with open('{}/{}.{}'.format(WORD_GRP, d, 'onion'), 'w', encoding='utf-8') as ff:
                for wrd in [w for w in wrdgrp if w != '']:
                    ff.write('{}\n'.format(wrd))   


def get_name(i):
    if i < 9:
        return '1000'+str(i)
    elif i >= 10 and i < 100:
        return '100'+str(i)
    else:
        return '10'+str(i)

def get_wordgrp(url_csv, wrdgrp_dir, proxies):
    if os.path.exists(wrdgrp_dir):
        process = subprocess.call(['rm', '-rf', wrdgrp_dir])
    else:
        os.makedirs(wrdgrp_dir)
    with open(url_csv, 'r', encoding='utf-8') as f:
        for i, url in enumerate(f):
            try:
                text = getOnionText(url, proxies=proxies)
            except KeyboardInterrupt:
                exit(-1)

            if text:
                with open('tmp.csv', 'w') as f:
                    f.write(text)

                process = subprocess.Popen(['java', 'GenerateWordGrp', 'tmp.csv'], stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                wrdgrp = stdout.decode('UTF-8').split('\n')
                with open('{}/{}.{}'.format(wrdgrp_dir, get_name(i), 'onion'), 'w', encoding='utf-8') as ff:
                    for wrd in [w for w in wrdgrp if w != '']:
                        ff.write('{}\n'.format(wrd)) 
    process = subprocess.call(['rm', '-rf', 'tmp.csv'])


 

                


if __name__ == '__main__':
    url_csv = sys.argv[1]

    proxies = {
    'http': 'socks5h://localhost:9050',
    'https': 'socks5h://localhost:9050'
    }

    WORD_GRP = sys.argv[2]
    
    get_wordgrp(url_csv, WORD_GRP, proxies=proxies)

