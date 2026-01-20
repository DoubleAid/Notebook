# conda 常见命令

## 创建虚拟环境

```bash
conda create -name <my-env>
conda create --name <my-env> python=3.9
conda env create -f environment.yml
```

yaml文件可以参考

```yaml
name: stats2
channels:
  - javascript
dependencies:
  - python=3.9
  - bokeh=2.4.2
  - conda-forge::numpy=1.21.*
  - nodejs=16.13.*
  - flask
  - pip
  - pip:
    - Flask-Testing
```

## 删除环境

```bash
conda remove -n <env-name> --all
```

+ **-n**: 指定环境名称（name 的缩写）；
+ **--all**: 删除该环境下的所有包和环境本身（必须加，否则只删包不删环境）。

## 查看环境列表

```bash
conda env list
conda info --envs
```

## 激活环境

```bash
conda activate <env-name>
```