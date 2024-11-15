import requests as reqs #type: ignore

BASE_URL = 'https://en.wikipedia.org/w/api.php'
DEFAULT_PARAMS = {'action': 'query', 'prop': 'extracts', 'format': 'json'} 

def wiki_api(**params) -> reqs.Response | None:
    """
        Wikipedia api : Utilisé pour effectuer des réquêtes
    """
    request_params = DEFAULT_PARAMS.copy()
    request_params.update(params)
    try:
        res = reqs.get(BASE_URL, request_params, timeout=10)
        res.raise_for_status()
        return res
    except reqs.exceptions.RequestException as e:
        print(f"(wiki-api)[Error]: {e}")
    return None
