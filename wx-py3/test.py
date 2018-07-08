import urllib.request
import urllib.parse
import ssl
import re

input = "Tsinghua 2012 Jie Tang"
context = ssl._create_unverified_context()
output = urllib.request.urlopen(
                    "https://tagging.aminer.cn/query/" + input,
                    context=context).read().decode(encoding='utf-8')

print(output)

dic = re.findall("\"([^\"]*)\": \"([^\"]*)\"", output)
ans = ""
for info in dic:
    ans += urllib.parse.quote(info[1]) + "=" + urllib.parse.quote(info[0]) + "&"
ans = ans[:-1]
ans = "https://www.aminer.cn/search?" + ans
print(ans)
