![](./extra_images/v2-1846948dbfbe33d410644c80b614cd50_1440w.jpg)

KMP算法
```c++
vector<int> get_kmp(const string& p) {
    vector<int> pmt(p.length(), 0);
    for (int i = 1, j = 0; i < p.length(); ++i) {
        while (j && p[i] != p[j]) j = pmt[j - 1];
        bool b = p[i] == p[j], c = p[i + 1] == p[j + 1];
        if (b) pmt[i] = pmt[j++];
        if (!b || !c) pmt[i] = j;
    }
    return pmt;
}

vector<int> kmp(const string& s, const string& p) {
    vector<int> pmt = get_kmp(p);
    vector<int> ans;
    for (int i = 0, j = 0; i < s.length(); ++i) {
        while (j && s[i] != p[j]) j = pmt[j - 1];
        if (s[i] == p[j]) j++;
        if (j == p.length()) {
            ans.emplace_back(i-j+1);
            j = pmt[j - 1];
        }
    }
    return ans;
}
```