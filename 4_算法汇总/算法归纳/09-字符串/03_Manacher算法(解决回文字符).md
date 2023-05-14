其中 r 为 回文的最大半径
```c++
constexpr auto MAXN = (int)7000000;

char s[MAXN << 1], str[MAXN];
int RL[MAXN];

int Manacher(void) {
	size_t len = strlen(str); 
    *s = '#';
	for (int i = 0; i < len; i++) {
		s[(i << 1) + 1] = str[i]; s[(i << 1) + 2] = '#';
	}
    len = (len << 1) + 1;

	int max = 1, pos, maxRight = -1; 
    memset(RL, 0, sizeof(RL));
	for (int i = 0; i < len; i++) {
		if (i < maxRight) RL[i] = std::min(RL[(pos << 1) - i], maxRight - i);
		else RL[i] = 1;
		while (i - RL[i] >= 0 && i + RL[i] < len && s[i - RL[i]] == s[i + RL[i]])++RL[i];
		if (i + RL[i] > maxRight) { pos = i; maxRight = i + RL[i]; }
		max = std::max(max, RL[i] - 1);
	} 
    return max;
}

# a # b # b # c # b #
maxR
```