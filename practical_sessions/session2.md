# Git


## Hands on
Version control system.

```
$ mkdir dev/project
$ cd dev/project
$ git init
Initialized empty Git repository in /Users/marco/dev/project/.git/
(master) $ 
```

A hidden folder `.git` appeared in the directory and everything you do in this directory will be recorded.

The name `(master)` (or `main`) that appeared near the prompt tells on which `branch` you're in at the moment.

Let's add something. Create a file `README` in the directory with some text inside.

```
(master) $ git status
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	README

nothing added to commit but untracked files present (use "git add" to track)
(master) $ git add README
(master) $ git commit -m "first commit"
[master (root-commit) 10a4880] first commit
 1 file changed, 1 insertion(+)
 create mode 100644 README
(master) $
```

Things to ponder:
   * tracked / untracked files
   * `add` specify what to track (or *add* to a tracking tree, `stage`)
   * `commit` save the tracking tree
   * each `commit` gives you a hash of the *node* in the tracking tree

To check history: `git log`.


### 2.1 Exercise: add more files, commit and explore the status of your working tree.

At some point you may want to explore some other ideas within your project.
That's a good place where start a new branch.

```
(master) $ git checkout -b newidea
Switched to a new branch 'newidea'
(newidea) $
```

You branched out from the `master` branch and started a new development branch.

```
(newidea) $ git branch
  master
* newidea
(END)
```

Press `q` to exit this prompt.

### 2.2 Exercise: add a file, commit and check the logs.

To have a glimpse of whats happening (`-s` flag):
```
(master) $ git status -s
```

## Merge

If you happen to work on a different branch, you may want at some point merge
the updates into another branch (tipically `master`).

### Case 1: easy

No updates were made on the `master` branch.

```
(newidea) $ git status
On branch newidea
nothing to commit, working tree clean
(newidea) $ git checkout master
(master) $ git merge newidea
Updating bc8f714..7482014
Fast-forward
 newideafile.txt | 7 +++++++
 1 file changed, 7 insertions(+)
 create mode 100644 mm.txt
(master) $
```

The two branched are now merged. `HEAD` points at the same commit hash.

### Case 2: hard

Updates were made on the `master` branch, maybe on the very same file.
The `merge` operation tries to do its best to merge data from different branches, but
sometimes it needs help.

```
(master) $ git merge newidea 
Auto-merging tempfile.txt
CONFLICT (content): Merge conflict in tempfile.txt
Automatic merge failed; fix conflicts and then commit the result.
(master) $ 
```

If you open the `tempfile.txt` (the file modified in both branch) you'll see
the differences.

For example:

```
<<<<<<< HEAD
itempfile.txt

some from master

=======
tempfile.txt

updates from newidea
>>>>>>> newidea
```

You have to manually fix the conflict (and delete the lines with `>>>>>>>`). Then commit the new `master` version.

```
(master) $ vi tempfile.txt
(master) $ git add tempfile.txt
(master) $ git commit -m "fixed"
(master) $
```

In alternative, you can force the update either from *our* version (in this case `master`) or 
from *their* version (in this case `newidea`)

```
(master) $ git checkout --ours -- tempfile.txt
(master) $ git add tempfile.txt
(master) $ git commit -m "merged our version"
```

```
(master) $ git checkout --theirs -- tempfile.txt
(master) $ git add tempfile.txt
(master) $ git commit -m "merged their version"
```

## History navigation

```
(master) $ git log --pretty=format:"%h %s"
442b72c Added feature
6c4b4c4 Typo
9c750fe New class customer
1a16fdc 1st commit
```

What was the status at commit `9c750fe`?

```
(master) $ git checkout 9c750fe .
(9c750fe) $
```

You're now in **DETACHED HEAD** state. 
You can do pretty much anything here (even commit updates), without affecting the working tree, read the instructions that are shown after you type the command.

When you're done:

```
(9c750fe) $ git reset --hard
HEAD is now at 9c750fe Add files via upload
(9c750fe) $ git checkout master
Previous HEAD position was 9c750fe Add files via upload
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
(master) $ 
```


## You really messed up?
[Look here](https://ohshitgit.com/) 




