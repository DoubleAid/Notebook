# 报错信息
```
rpm: RPM should not be used directly install RPM packages, use Alien instead!
rpm: However assuming you know what you are doing...
warning: bcompare-4.4.3.26655.x86_64.rpm: Header V4 DSA/SHA1 Signature, key ID 7f8840ce: NOKEY
```

# 解决方法
```bash
# Install alien and all the dependencies it needs
apt-get install alien dpkg-dev debhelper build-essential
# convert a package from rpm to debian format
alien packagen.rpm
# install package
dpkg -i package.deb
```