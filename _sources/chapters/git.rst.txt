
Delete everything and restore what you want
-------------------------------------------

Credit where credit is due: 
https://stackoverflow.com/a/17909526/13239311

Rather than delete this-list-of-files one at a time, do the almost-opposite: 
delete everything and just restore the files you want to keep.

Like so:

.. code:: bash

    # for unix

    git checkout master
    git ls-files > keep-these.txt
    git filter-branch --force --index-filter \
        "git rm  --ignore-unmatch --cached -qr . ; \
        cat $PWD/keep-these.txt | tr '\n' '\0' | xargs -d '\0' git reset -q \$GIT_COMMIT --" \
        --prune-empty --tag-name-filter cat -- --all

.. code:: bash

    # for macOS

    git checkout master
    git ls-files > keep-these.txt
    git filter-branch --force --index-filter \
        "git rm  --ignore-unmatch --cached -qr . ; \
        cat $PWD/keep-these.txt | tr '\n' '\0' | xargs -0 git reset -q \$GIT_COMMIT --" \
        --prune-empty --tag-name-filter cat -- --all


Cleanup
+++++++


.. code:: bash

    rm -rf .git/refs/original/
    git reflog expire --expire=now --all
    git gc --prune=now

    # optional extra gc. Slow and may not further-reduce the repo size
    git gc --aggressive --prune=now