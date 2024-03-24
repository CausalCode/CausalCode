# -*- coding: utf-8 -*-
# author:yejunyao
# datetime:2023/10/24 20:29

"""
description：
"""



import git
def commit(content):
    repo = git.Repo(search_parent_directories=True)
    try:
        g = repo.git
        g.add("--all")
        res = g.commit("-m " + content)
        print(res)
    except Exception as e:
        print("no need to commit")
