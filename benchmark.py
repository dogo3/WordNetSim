"""
Benchmark to evaluate WNS performance in your system.

Execute it with: python3 benchmark.py n_iterations

Where n_iterations is the number of times each method will be
executed for computing the computing time average.
Example:
    python3 benchmark.py 10
"""

import sys
import time
print("Importing wns",flush=True)
start_time = time.time()
from WNS import Lemmatizer,sim_str_str,sim_str_str_multiling, toks_to_synsets, similarity_score, symetric_similarity_score, sim_str_attrlst,sim_str_attrlst_multiling
print("Time to import WNS",time.time() - start_time,"s")

n_iterations = 1
if len(sys.argv)==2:
    n_iterations = int(sys.argv[1])

#wc quijote.txt = 1876 words
txt = open("quijote.txt","r").read()
print("START PROGRAM",flush=True)
start_time = time.time()
for i in range(n_iterations):
    lemmatizer = Lemmatizer(LANG="es",LANG_STOPWORDS="spanish")
print("Time to create Lemmatizer",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    lemmas = lemmatizer.lemmatize(txt)
end_time = time.time()
print("Time to lemmatize paragraph",(end_time - start_time)/n_iterations,"s")
print("    that is",((end_time - start_time)/n_iterations)/len(lemmas),"seconds per lemma.")

#We measure how fast is synset recognition and path measurement
start_time = time.time()
for i in range(n_iterations):
    toks_to_synsets(lemmas,lang="spa")
end_time = time.time()
print("Time to find synsets of the paragraph",(end_time - start_time)/n_iterations,"s")
print("    that is",((end_time - start_time)/n_iterations)/len(lemmas),"seconds per lemma.")

toks1 = ["Hola","llamar","casa"]
s1 = toks_to_synsets(toks1,lang="spa")
toks2 = ["Adiós","volver","hotel"]
s2 = toks_to_synsets(toks2,lang="spa")

# Measuring internal path similarity methods
start_time = time.time()
for i in range(n_iterations):
    symetric_similarity_score(s1,s2,stat="max")
end_time = time.time()
print("Time to measure similarity",(end_time - start_time)/n_iterations,"s")
print("    that is",((end_time - start_time)/n_iterations)/(len(toks1)*len(toks1)),"seconds per pair of synsets.")

# Measuring API methods
start_time = time.time()
for i in range(n_iterations):
    sim_str_str("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                        "Mi ordenador no funciona.",\
                        lang1="es",\
                        lang2="es",\
                        stat="max")

print("Avg time WNS without language inference (es)",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_str("I am testing this new application in a laptop connected to the internet.",
                        "My laptop is not working.",\
                        lang1="en",\
                        lang2="en",\
                        stat="max")

print("Avg time WNS without language inference (en)",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_str("我正在一台连接到互联网的笔记本电脑中测试这个新的应用程序。",
                        "我的笔记本电脑不工作了。",\
                        lang1="zh",\
                        lang2="zh",\
                        stat="max")

print("Avg time WNS without language inference (zh)",(time.time() - start_time)/n_iterations,"s")


start_time = time.time()
for i in range(n_iterations):
    sim_str_str_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                        "Mi ordenador no funciona.",\
                        stat="max")
print("Avg time WNS with language inference",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_str("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                        "Mi ordenador no funciona.",\
                        lang1="es",\
                        lang2="es",\
                        stat="max")

print("Avg time WNS without language inference",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_str_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",
                        "Mi ordenador no funciona.",\
                        stat="max")
print("Avg time WNS with language inference",(time.time() - start_time)/n_iterations,"s")


start_time = time.time()
for i in range(n_iterations):
    sim_str_attrlst("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",[("profesión","programador aplicaciones")],lang1="es",lang2="es",stat="max")
print("Avg time WNS between attribute list (1) and text without language inference",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_attrlst("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",[("profesión","programador aplicaciones"),("vehículo","coche"),("afición","escalada"),("estudios superiores","ingeniería informática")],lang1="es",lang2="es",stat="max")
print("Avg time WNS between attribute list (4) and text without language inference",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_attrlst_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",[("work","applications programmer")],stat="max")
print("Avg time WNS between attribute list (1) and text with language inference",(time.time() - start_time)/n_iterations,"s")

start_time = time.time()
for i in range(n_iterations):
    sim_str_attrlst_multiling("Estoy probando esta nueva aplicación en mi portátil conectado a internet.",[("work","applications programmer"),("vehicle","car"),("hobby","climbing"),("superior studies","computer science")],stat="max")
print("Avg time WNS between attribute list (4) and text with language inference",(time.time() - start_time)/n_iterations,"s")
