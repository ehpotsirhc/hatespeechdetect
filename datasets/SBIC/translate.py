#!/usr/bin/env python3

# Translation Script
# Christophe Leung
# October 30, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# =============================================================================
from pathlib import Path
from datetime import datetime
import sys, logging, os, random, time, json
import pandas as pd
import googletrans

# =============================================================================

class Constants:
    FPATH_DATA = Path('SBIC.v2.agg.cmb_processed.csv')
    DPATH_LOGS = Path('translated')
    FPATH_LOG_MAIN = Path('translate.log')
    FPATH_OUTPUT = Path('translated')
    FPATH_SESSION = DPATH_LOGS/'session.json'
    # LANGUAGES = ['chinese (simplified)', 'hindi', 'spanish', 'french', 'russian', 'japanese', 'german']
    LANGUAGES = ['hindi', 'spanish', 'russian']
    DELAY_THRESHOLD = 500
    T_DELAY_INTERIM = 100
    T_DELAY_RETRY = 10


class Utils:
    @staticmethod
    def logging_init():
        logpath = Constants.DPATH_LOGS/Constants.FPATH_LOG_MAIN
        os.makedirs(logpath.parent) if not logpath.parent.exists() else None
        logging.basicConfig(
            filename=logpath, 
            filemode='a', 
            format='%(asctime)s - [%(levelname)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S', 
            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        return logging


class Persistence:
    @staticmethod
    def get(slot):
        if Constants.FPATH_SESSION.exists():
            with open(Constants.FPATH_SESSION, 'r') as f:
                return json.load(f)[slot]

    @staticmethod
    def put(slot, token):
        session = {'current':'', 'completed':[]}
        if Constants.FPATH_SESSION.exists():
            with open(Constants.FPATH_SESSION, 'r') as f:
                session = json.load(f)
        if slot=='completed':
            session[slot].append(token)
        elif slot=='current':
            session[slot]=token
        with open(Constants.FPATH_SESSION, 'w') as f:
            json.dump(session, f, indent=4)


class Translate:
    def __init__(self):
        self.translator = googletrans.Translator()
        self.counter = 0

    def languages_available(self):
        return googletrans.LANGUAGES
    
    def languages_validate(self, target_languages):
        available = googletrans.LANGUAGES.items()
        return {lang_key:lang_name for (lang_key, lang_name) in available if lang_name in target_languages}
    
    def translate(self, text, src, dst, **kwargs):
        delay_rand = Constants.T_DELAY_INTERIM + random.randint(0, 15)     # prevent API timeouts
        retries = kwargs['retry'] if 'retry' in kwargs else 3
        try:
            if self.counter > Constants.DELAY_THRESHOLD:
                logging.info('Manually-induced API delay; Next translation will start in %s seconds...' % (delay_rand))
                time.sleep(delay_rand)
                self.counter = 0
            self.counter += 1
            translation = self.translator.translate(text, src=src, dest=dst)
            return translation.text
        except Exception as error:
            if retries > 0:
                timeout = Constants.T_DELAY_RETRY + random.randint(3, 10)
                logging.warning('Translation error (%s -> %s). Retrying in %s seconds...' % (src, dst, timeout))
                time.sleep(timeout)
                retries -= 1
                self.translate(text, src, dst, retry=retries)
            else:
                raise error


def main():
    Utils.logging_init()
    translate = Translate()
    dst_valid = translate.languages_validate(Constants.LANGUAGES)
    stats_nrequested, stats_nvalid = len(Constants.LANGUAGES), len(dst_valid)
    stats_resolved = ',  '.join(map(lambda lang: '%s:%s' % (lang[0], lang[1]), dst_valid.items()))
    df_data = pd.read_csv(Constants.FPATH_DATA)
    
    print('Languages Available...')
    print(translate.languages_available(), '\n')
    
    logging.info('Translating data within "%s"...' % Constants.FPATH_DATA)
    logging.info('Requested Languages: [%s]' % ', '.join(sorted(Constants.LANGUAGES)))
    logging.info('Resolvable Languages: (%s/%s) [%s]' % (stats_nvalid, stats_nrequested, stats_resolved))

    for i, dst in enumerate(dst_valid):
        if dst not in Persistence.get('completed'):
            logging.info('Beginning translation for target language "%s"...' % dst)
            Persistence.put('current', dst)
            src, translated_text = 'en', []
            fpath_translated = Constants.FPATH_OUTPUT/('translated_%s.csv' % dst)
            translated_text = pd.read_csv(fpath_translated).values.tolist() if fpath_translated.exists() else translated_text
            for idx, (text, label_name, label_id) in df_data.iterrows():
                msg_head = ('[translating %s, %s -> %s]' % (str(idx+1).zfill(len(str(len(df_data)))), src, dst))
                if idx < len(translated_text):
                    logging.info('%s Already translated. Skipping.' % msg_head)
                    continue
                translation = translate.translate(text, src, dst)
                logging.info('%s "%s" -> "%s"' % (msg_head, text, translation))
                translated_text.append((translation,label_name,label_id))
                df_translated_row = pd.DataFrame(translated_text, columns=df_data.columns)
                df_translated_row.to_csv(fpath_translated, index=False)
            logging.info('Translation for "%s" complete.' % dst)
            Persistence.put('completed', dst)
            Persistence.put('current', '')
        else:
            Persistence.put('current', '')
            logging.info('Destination language "%s" already fully translated. Skipping.' % dst)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        Constants.FPATH_DATA = Path(sys.argv[1])
    main()

