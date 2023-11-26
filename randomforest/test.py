from urllib import parse
# params = '90'
# print(parse.quote('三大法人買賣張數'))
# for i in ['aaa', 'bbb']:
#     print(i)


from sklearn.metrics import precision_score, accuracy_score
a = [1,0,1,1,1,0,1]
b = [1,0,1,1,1,0,0]
print(precision_score(a,b))
print(accuracy_score(a,b))