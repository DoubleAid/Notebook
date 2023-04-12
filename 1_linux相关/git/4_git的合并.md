
## 合并
比如 我们在 master 分支上 创建了一个 cat 分支， 并提交了两次修改，现在需要把cat分支上的修改合并到 master 上

```bash
# 首先切换会 master 分支
git checkout master
# 使用 merge 指令合并分支
git merge cat
```

## 恢复删除的分支修改
比如 我们在 master 分支上 创建了一个 cat 分支， 并提交了两次修改，之后删除了cat分支
现在需要把cat分支上的修改合并到 master 上
```bash
# 删除分支
git branch -d cat
git branch -D cat

# 根据 commit 创建新的分支
git branch new_cat b174a5a
git checkout new_cat
```

## rebase 合并
有两个分支基于 master 分支， 分别是 cat 和 dog， 现在想要合并

```bash
git check cat
git rebase dog
```
原cat分支下的 commit 会作废， 会在dog分支下差生两次新的提交， 并把dog分支提交后的分支作为cat分支

如何取消rebase
```bash
# 使用 reflog 
git log --oneline #查看日志
git reflog # 找到rebase提交之前最后一个commit的hash值
git reset b174a5a --hard
```

## 解决分支冲突
有两个分支基于 master 分支， 分别是 cat 和 dog， 两者都修改了animal文件，现在想要合并

无论是使用merge 或者 rebase 都会产生冲突
```bash
# 使用 merge
git merge dog
auto-merging animal
CONFLICT(content): Merge conflict in index.html
```
git 发现 animal 文件有问题
```shell
# 查看状态
git status
# 手动修改相应的变化
git add animal
git commit -m "conflict fixed"
```

当使用 rebase 时
```shell
git rebase dog
git status 
git add animal
```

## 如果是图片或者其他的怎么办
上面的 animal 因為是文字檔案，所以 Git 可以標記出發生衝突的點在哪些行，我們用肉眼都還能看得出來大
、概該怎麼解決，但如果是像圖片檔之類的二進位檔怎麼辦？例如在 cat 分支跟 dog 分支，同時都加了一張叫做
cute_animal.jpg 的圖片，合併的時候出現衝突的訊息：
```shell
$ git merge dog
warning: Cannot merge binary files: cute_animal.jpg (HEAD vs. dog)
Auto-merging cute_animal.jpg
CONFLICT (add/add): Merge conflict in cute_animal.jpg
Automatic merge failed; fix conflicts and then commit the result.
```
討論後決定貓才是這世上最可愛的動物，所以決定要用 cat 分支的檔案：
```shell
git checkout --ours cute_animal.jpg
```
如果是要用對方（dog 分支），則是使用 --theirs 參數
```shell
$ git checkout --theirs cute_animal.jpg
```
決定之後，就跟前面一樣，加到暫存區，準備 Commit，然後結束這一回合。