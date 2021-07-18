from WNS import fill_scores
import marshal

f = open("txts/en.txt")
txt_en = f.read()
fill_scores(txt_en,"en")

f = open("./scores.marshal","rb")
scores = marshal.load(f)
print(list(scores.keys())[0:10])
print(len(scores))