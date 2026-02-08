import re
from urllib.parse import urlparse
import pandas as pd


def extract_features(df):

    def having_ip_address(url):
        return 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    def abnormal_url(url):
        hostname = urlparse(url).hostname
        return 1 if hostname and hostname in url else 0

    def google_index(url):
        # Dummy safe placeholder (no external dependency)
        return 1 if len(url) % 2 == 0 else 0

    def shortening_service(url):
        return 1 if re.search(
            r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly', url
        ) else 0

    def suspicious_words(url):
        return 1 if re.search(
            r'login|signin|bank|account|update|free|bonus|paypal|verify', url, re.I
        ) else 0

    def fd_length(url):
        try:
            return len(urlparse(url).path.split('/')[1])
        except:
            return 0

    def tld_length(url):
        try:
            return len(urlparse(url).netloc.split('.')[-1])
        except:
            return 0

    # === Feature columns ===
    df['use_of_ip'] = df['URL'].apply(having_ip_address)
    df['abnormal_url'] = df['URL'].apply(abnormal_url)
    df['google_index'] = df['URL'].apply(google_index)
    df['count_dot'] = df['URL'].apply(lambda x: x.count('.'))
    df['count-www'] = df['URL'].apply(lambda x: x.count('www'))
    df['count@'] = df['URL'].apply(lambda x: x.count('@'))
    df['count_dir'] = df['URL'].apply(lambda x: urlparse(x).path.count('/'))
    df['count_embed_domain'] = df['URL'].apply(lambda x: urlparse(x).path.count('//'))
    df['short_url'] = df['URL'].apply(shortening_service)
    df['count-https'] = df['URL'].apply(lambda x: x.count('https'))
    df['count-http'] = df['URL'].apply(lambda x: x.count('http'))
    df['count%'] = df['URL'].apply(lambda x: x.count('%'))
    df['count?'] = df['URL'].apply(lambda x: x.count('?'))
    df['count-'] = df['URL'].apply(lambda x: x.count('-'))
    df['count='] = df['URL'].apply(lambda x: x.count('='))
    df['url_length'] = df['URL'].apply(len)
    df['hostname_length'] = df['URL'].apply(lambda x: len(urlparse(x).netloc))
    df['sus_url'] = df['URL'].apply(suspicious_words)
    df['fd_length'] = df['URL'].apply(fd_length)
    df['tld_length'] = df['URL'].apply(tld_length)
    df['count-digits'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in x))
    df['count-letters'] = df['URL'].apply(lambda x: sum(c.isalpha() for c in x))

    return df
