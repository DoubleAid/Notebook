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

## 查看环境列表

```bash
conda env list
conda info --envs
```

## 激活环境

```bash
conda activate <env-name>
```