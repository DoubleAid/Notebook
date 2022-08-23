### <font color=deepskyblue>快速索引</font>
- [<font color=deepskyblue>快速索引</font>](#font-colordeepskyblue快速索引font)
- [安装](#安装)
- [下载分支](#下载分支)
- [<font color=deepskyblue>git提交本地代码到远程分支</font>](#font-colordeepskybluegit提交本地代码到远程分支font)
- [<font color=deepskyblue>修改远端仓库</font>](#font-colordeepskyblue修改远端仓库font)
- [<font color=deepskyblue>更新本地分支</font>](#font-colordeepskyblue更新本地分支font)
### 安装
```
sudo -i
apt-get install git
```

### 下载分支
```
git clone -b v5/map-pipeline git@****/common.git
```

### <font color=deepskyblue>git提交本地代码到远程分支</font>
```
git init # 建立本地仓库
git add .
git commit -m "new branch commit"
git remote add origin git@xxxxxx.git
git branch hello_new_branch
git checkout hello_git_branch
git push origin hello_git_branch
```
### <font color=deepskyblue>修改远端仓库</font>
+ 方法一 通过命令直接修改远程地址
```
进入git_test根目录
git remote 查看所有远程仓库， git remote xxx 查看指定远程仓库地址
git remote set-url origin http://192.168.100.235:9797/john/git_test.git
```
+ 方法二 通过命令先删除再添加远程仓库
```
进入git_test根目录
git remote 查看所有远程仓库， git remote xxx 查看指定远程仓库地址
git remote rm origin
git remote add origin http://192.168.100.235:9797/john/git_test.git
```
方法三 直接修改配置文件
```
进入git_test/.git

vim config

[core]
repositoryformatversion = 0
filemode = true
logallrefupdates = true
precomposeunicode = true
[remote "origin"]
url = http://192.168.100.235:9797/shimanqiang/assistant.git
fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
remote = origin
merge = refs/heads/master

修改 [remote “origin”]下面的url即可
```

### <font color=deepskyblue>更新本地分支</font>
1. 更新远程文件到本地方式一
    1. 查看远程仓库  
    `git remote -v`
    2. 从远程获取最新版本到本地  
    `git fetch origin aaa`
    3. 比较远程分支和本地分支  
    `git log -p aaa origin/aaa`
    4. 合并远程分支到本地  
    `git merge origin/aaa`

2. 远程文件到本地方式二，在本地建临时分支，合并后删除
    1. 查看远程仓库  
    `git remote -v`
    2. 从远程获取最新版本到本地  
    `git fetch origin master:temp`
    3. 比较远程分支和本地分支的区别  
    `git diff temp`
    4. 合并远程分支到本地  
    `git merge temp`

3. 使用pull更新  
    `git pull origin aaa`