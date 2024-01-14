#coding=UTF-8



from django.shortcuts import render,HttpResponse
import json
import time
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from  django.utils.http  import  urlquote,unquote
from. import InformationRetrival as ir

s = ir.Searcher()

def index(request):
    return render(request,'index.html')

class home(View):
    def get(self,request):
         return render(request,'home.html')


class result(View):
    def get(self,request,query,tag):
        # with open('static/11.txt','w',encoding='utf-8') as f:
        #     f.write(query)
        print(unquote(unquote(query, encoding='utf-8'), encoding='utf-8'))
        searchquery=unquote(unquote(query, encoding='utf-8'), encoding='utf-8')
        tags=tag.split('_')
        ordermethod = "0"
        minprice = '0'
        maxprice = '99999'

        #provide searchquery:str,tags:list,ordermethos:str(default=0),minprice:str(default=0),maxprice:str(default=99999) return recommendlist:list,goodslist:list
        goodslist, recommendlist = \
        s.run(search_text=searchquery,
              recommend_list=[],
              tag=tags,
              order=ordermethod,
              min_price=minprice,
              max_price=maxprice
              )

        # tags=json.dumps(tags,ensure_ascii=False)
        # recommendlist=json.dumps(recommendlist,ensure_ascii=False)
        # goodslist=json.dumps(goodslist,ensure_ascii=False)
        return render(request,'searchresult.html',{'query':searchquery,'tags':tags,'recommendlist':recommendlist,
                                                   'items':goodslist})
    def post(self,request,query,tag):
        searchquery=query
        tags=tag.split('_')
        entrance=request.POST.get('entrance')
        ret={'status':True,'message':None}
        if entrance=='order':
            #TODO:
            ordermethod=request.POST.get('method')
            minprice = '0'
            maxprice = '99999'
            goodslist, recommend_list = \
            s.run(search_text=searchquery,
              recommend_list=[],
              tag=tags,
              order=ordermethod,
              min_price=minprice,
              max_price=maxprice
              )
            #provide searchquery:str,tags:list,ordermethos:str(1,2,3,4,5,6),minprice:str(default=0),maxprice:str(default=99999)
            # return recommendlist:list,goodslist:list
            ret['message'] = goodslist
            return HttpResponse(json.dumps(ret))
        elif entrance == 'price':
            ordermethod=request.POST.get('method')
             #provide searchquery:str,tags:list,ordermethos:str(1,2,3,4,5,6),minprice:str(default=0),maxprice:str(default=99999)
            # return: recommendlist:list,goodslist:list
            minprice= request.POST.get('minprice')
            maxprice= request.POST.get('maxprice')
            goodslist, recommend_list = \
            s.run(search_text=searchquery,
              recommend_list=[],
              tag=tags,
              order=ordermethod,
              min_price=minprice,
              max_price=maxprice
              )
            #provide searchquery:str,tags:list,ordermethos:str(1,2,3,4,5,6),minprice:str(default=0),maxprice:str(default=99999)return recommendlist:list,goodslist:list
            ret['message'] = goodslist
            return HttpResponse(json.dumps(ret))
        elif entrance == 'pic':
            goods={}
            sku_id= request.POST.get('sku_id')
            print(sku_id)
            #provide: sku_id:str
            # return ciyunpath,datasandian,dataradar
            sku_id = int(sku_id)
            ciyunpath = s.get_word_cloud(sku_id)
            datasandian = s.get_coordinate(sku_id)
            dataradar = s.get_radar(sku_id)
            goods['ciyunpath']=ciyunpath
            goods['datasandian']=datasandian
            print(datasandian)
            goods['dataradar']=dataradar
            ret['message']=goods
            return HttpResponse(json.dumps(ret))

