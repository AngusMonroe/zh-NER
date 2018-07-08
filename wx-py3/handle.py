# -*- coding: utf-8 -*-
# filename: handle.py
import hashlib
import reply
import receive
import web
import urllib.request
import ssl
class Handle(object):
    def POST(self):
        try:
            webData = web.data()
            print("Handle Post webdata is ", webData)  # 后台打日志
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg) and recMsg.MsgType == 'text':
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                input = recMsg.Content.decode(encoding='utf-8')
                context = ssl._create_unverified_context()
                output = urllib.request.urlopen(
                    "https://tagging.aminer.cn/query/" + input,
                    context=context).read().decode(encoding='utf-8')
                print(type(output))
                replyMsg = reply.TextMsg(toUser, fromUser, output)
                return replyMsg.send()
            else:
                print("暂且不处理")
                return "success"
        except Exception as Argment:
            return Argment
