
Store your dotfiles
-------------------

This is taken partially from https://www.atlassian.com/git/tutorials/dotfiles .
I simply moved it here in case ``Atlassian`` should stop existing at any point
in time. 

We store a Git bare repository in a "side" folder (like $HOME/.cfg or
$HOME/.myconfig) using an alias so that commands are run against that repository
and not the usual ``.git`` local folder.

.. code:: bash

    git init --bare $HOME/.cfg
    alias config='/usr/bin/git --git-dir=$HOME/.cfg/ --work-tree=$HOME'
    config config --local status.showUntrackedFiles no
    echo "alias config='/usr/bin/git --git-dir=$HOME/.cfg/ --work-tree=$HOME'" >> $HOME/.bashrc

Line-by-Line

1. The first line creates a folder ~/.cfg which is a Git bare repository that
   will track our files. 
2. Then we create an alias config which we will use
   instead of the regular git when we want to interact with our configuration
   repository. 
3. We set a flag - local to the repository - to hide files we are
   not explicitly tracking yet. This is so that when you type config status and
   other commands later, files you are not interested in tracking will not show up
   as untracked. 
4. Also you can add the alias definition by hand to your .bashrc
   or use the the fourth line provided for convenience.



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