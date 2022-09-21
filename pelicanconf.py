#!/usr/bin/env python
# -*- coding: utf-8 -*- #
AUTHOR = 'ThinkNotClear'
SITENAME = 'ThinkNotClear'
# SITEURL = 'papernotclear'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Asia/Shanghai'

DEFAULT_LANG = 'zh'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

THEME = r'E:\project\papernotclear\pelican\pelican-elegant'
PLUGIN_PATHS = [r'E:\project\papernotclear\pelican\pelican-plugins']
PLUGINS = ['sitemap', 'extract_toc', 'obsidian', 'tipue_search']  #, 'ipynb.liquid']
# MD_EXTENSIONS = ['codehilite(css_class=highlight)', 'extra', 'headerid', 'toc', 'problem']
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
        'markdown.extensions.toc': {'title': 'Table of Contents'},
    },
    'output_format': 'html5',
}

LANDING_PAGE_ABOUT = {'title': 'Think不Clear', 
        "details": '''蜻蜓点论文：<br>
        B站: <a href="https://space.bilibili.com/17529417" target="_blank">Bilibili</a><br>
        Youtube: <a href="https://www.youtube.com/channel/UCwyMgDylnGQ-Wf4N00RmWJw" target="_blank">频道主页</a><br>
        西瓜视频: <a href="https://www.ixigua.com/home/110050360886" target="_blank">西瓜</a><br>
        今日头条(全是视频): <a href="https://www.toutiao.com/c/user/token/MS4wLjABAAAAlpSRi3mwMIj0kwU4ZylX9sESVVhiICJ-LJTPwH2xv3w/" target="_blank">头条</a><br>

        <br>
        ThinkNotClear<br>
        PaperSkimn English: <br>
        Youtube: <a href="https://www.youtube.com/channel/UChaDVzxo8wkZwFYA0xso82w" target="_blank">Channel</a><br>
        '''}

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 20

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True