import wikipedia


def get_wiki_content():
    wikipedia.set_lang('en')
    page = wikipedia.page('Artificial intelligence')
    return page.summary
