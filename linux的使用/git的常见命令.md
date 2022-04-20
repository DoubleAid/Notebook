### <font color=deepskyblue>快速索引</font>
[提交本地代码到远程分支](#font-colordeepskybluegit提交本地代码到远程分支font)

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
