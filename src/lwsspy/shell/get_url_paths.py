import requests
from bs4 import BeautifulSoup


def get_url_paths(url: str, ext: str = '', params: dict = {}):
    """Gets file urls from a webserver.

    Parameters
    ----------
    url : str
        webserver url
    ext : str, optional
        extension, by default ''
    params : dict, optional
        parameters, by default {}

    Returns
    -------
    [type]
        [description]
    """

    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a')
              if node.get('href').endswith(ext)]
    return parent
