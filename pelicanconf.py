#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Derek Howard'
SITENAME = 'Derek Howard'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/Toronto'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

THEME = '../pelican-themes/pelican-bootstrap3'
BOOTSTRAP_THEME = 'simplex'

# Plugins
PLUGIN_PATHS = ['../pelican-plugins']
PLUGINS = ['neighbors', 'feed_summary']

FEED_USE_SUMMARY = True
SUMMARY_MAX_LENGTH = 50

#Choose to display category of posts on menu besides About/etc pages
DISPLAY_CATEGORIES_ON_MENU = False

# Blogroll
#LINKS = (('Pelican', 'http://getpelican.com/'),
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('Github', 'https://github.com/derekhoward'),
          ('Twitter', 'https://twitter.com/derekhoward0'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# Formatting for dates

DEFAULT_DATE_FORMAT = ('%a %d %B %Y')

# Formatting for urls

ARTICLE_URL = "posts/{date:%Y}/{date:%m}/{slug}/"
ARTICLE_SAVE_AS = "posts/{date:%Y}/{date:%m}/{slug}/index.html"

CATEGORY_URL = "category/{slug}"
CATEGORY_SAVE_AS = "category/{slug}/index.html"

TAG_URL = "tag/{slug}/"
TAG_SAVE_AS = "tag/{slug}/index.html"