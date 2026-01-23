import re
from urllib.parse import urlparse
from tld import get_tld
import pandas as pd


def extract_features(df):
    def having_ip_address(url):
        match = re.search(
            r'(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            r'([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            r'((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            return 1
        else:
            return 0


    def abnormal_url(url):
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        match = re.search(hostname, url)
        if match:
            return 1
        else:
            return 0

    def google_index(url):
        try:
          return 1 if len(url) % 2 == 0 else 0
        except:
          return 0

    def count_dot(url):
        count_dot = url.count('.')
        return count_dot
    
    def count_www(url):
        url.count('www')
        return url.count('www')
    
    def count_atrate(url):
        return url.count('@')
    
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')
    
    def no_of_embed(url):
        urldir = urlparse(url).path
        return urldir.count('//')

    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          r'tr\.im|link\.zip\.net',
                          url)
        if match:
            return 1
        else:
            return 0
        
    def count_https(url):
        return url.count('https')
    
    def count_http(url):
        return url.count('http')
    
    def count_per(url):
        return url.count('%')
    
    def count_ques(url):
        return url.count('?')
    
    def count_hyphen(url):
        return url.count('-')
    
    def count_equal(url):
        return url.count('=')
    
    def url_length(url):
        return len(str(url))#Length of URL
    
    def hostname_length(url):
        return len(urlparse(url).netloc)

    def suspicious_words(url):
        match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                          url)
        if match:
            return 1
        else:
            return 0
    
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    
    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters

    def get_tld(url):
        try:
          netloc = urlparse(url).netloc
          parts = netloc.split('.')
          if len(parts) > 1:
            return parts[-1]
          else:
            return ''
        except:
           return ''

    def fd_length(url):
        urlpath= urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0
    

    df['use_of_ip'] = df['URL'].apply(lambda i:having_ip_address(i))
    df['abnormal_url'] = df['URL'].apply(lambda i: abnormal_url(i))
    df['google_index'] = df['URL'].apply(lambda i: google_index(i))
    df['count_dot'] = df['URL'].apply(lambda i: count_dot(i))
    df['count-www'] = df['URL'].apply(lambda i: count_www(i))
    df['count@'] = df['URL'].apply(lambda i: count_atrate(i))
    df['count_dir'] = df['URL'].apply(lambda i: no_of_dir(i))
    df['count_embed_domain'] = df['URL'].apply(lambda i: no_of_embed(i))
    df['short_url'] = df['URL'].apply(lambda i: shortening_service(i))
    df['count-https'] = df['URL'].apply(lambda i : count_https(i))
    df['count-http'] = df['URL'].apply(lambda i : count_http(i))
    df['count%'] = df['URL'].apply(lambda i : count_per(i))
    df['count?'] = df['URL'].apply(lambda i: count_ques(i))
    df['count-'] = df['URL'].apply(lambda i: count_hyphen(i))
    df['count='] = df['URL'].apply(lambda i: count_equal(i))
    df['url_length'] = df['URL'].apply(lambda i: url_length(i))
    df['hostname_length'] = df['URL'].apply(lambda i: hostname_length(i))
    df['sus_url'] = df['URL'].apply(lambda i: suspicious_words(i))
    df['fd_length'] = df['URL'].apply(lambda i: fd_length(i))
    df['tld_length'] = df['URL'].apply(lambda i: get_tld(i))
    df['count-digits']= df['URL'].apply(lambda i: digit_count(i))
    df['count-letters']= df['URL'].apply(lambda i: letter_count(i))

    return df
