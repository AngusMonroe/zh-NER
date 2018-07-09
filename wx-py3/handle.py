# -*- coding: utf-8 -*-
# filename: handle.py
import hashlib
import reply
import receive
import web
import ssl
import urllib.request
import urllib.parse
import re
class Handle(object):
    def POST(self):
        try:
            webData = web.data()
            print ("Handle Post webdata is ", webData)
   #后台打日志
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg) and recMsg.MsgType == 'text':
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                input = recMsg.Content.decode(encoding='utf-8')
                context = ssl._create_unverified_context()
                output = urllib.request.urlopen(
                    "https://tagging.aminer.cn/query/" + input,
                    context=context).read().decode(encoding='utf-8')
                print(output)
                dic = re.findall("\"([^\"]*)\": \"([^\"]*)\"", output)
                url = "https://www.aminer.cn/search?"
                ans = ""
                for info in dic:
                    if info[1] == "org":
                        url += "o=" + urllib.parse.quote(info[0]) + "&"
                        ans += "来自" + info[0]
                for info in dic:
                    if info[1] == "name":
                        url += "n=" + urllib.parse.quote(info[0]) + "&"
                        ans += "名为" + info[0] + "的学者"
                for info in dic:
                    if info[1] == "item":
                        url += "k=" + urllib.parse.quote(info[0]) + "&"
                        ans += "的" + info[0]
                url = url[:-1]
                link = "<a href=\"" + url + "\">" + ans + "</a>"
                print(link)
                replyMsg = reply.TextMsg(toUser, fromUser, link)
                return replyMsg.send()
            else:
                print ("暂且不处理")
                return "success"
        except Exception as Argment:
            return Argment


